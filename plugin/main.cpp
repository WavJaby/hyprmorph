// Resize transition plugin, skeleton (S1-A): every window size change shows the old snapshot, then the new buffer, plain-scaled into the
// animated box, with a short crossfade. Replaces the native path's UV stretch/crop (gap on shrink, progressive crop on grow, F12).
//
// Generation trigger: GEOMETRIC_GOAL size change, cause-agnostic (dispatcher, togglefloating, relayout, fullscreen).
// Capture timing: m_realSize's begin callback only queues sendWindowSize via doLater, and m_events.resize fires on animation ticks, which may
// run inside renderMonitor. So the goal is checked (and the old snapshot captured) at the two points that precede any texture replacement
// and run outside a render pass: the window surface's precommit, and render.preChecks of every monitor.
// Each generation also holds the size/position animation for up to hold_ms (Hold.hpp, D3-3); the old/new switch waits for the release.
// Readiness (S5): thumbnails of the new-content capture feed a per-generation detector on a worker thread.
// Prep (S5): each readiness selection sends the old snapshot and the current new content, at analysis resolution, to the out-of-process
// prep worker (D1-1); its answer releases the hold and, when it came before the box moved, becomes the mesh the Presenter draws.
#include "Hold.hpp"
#include "Prep.hpp"
#include "Presenter.hpp"
#include "Readback.hpp"
#include "Readiness.hpp"
#include "Thumbnail.hpp"
#include "Transition.hpp"

#include <hyprland/src/plugins/PluginAPI.hpp>
#include <hyprland/src/config/values/ConfigValues.hpp>
#include <hyprland/src/debug/log/Logger.hpp>
#include <hyprland/src/desktop/state/WindowState.hpp>
#include <hyprland/src/desktop/view/Window.hpp>
#include <hyprland/src/event/EventBus.hpp>
#include <hyprland/src/managers/eventLoop/EventLoopManager.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/protocols/core/Compositor.hpp>
#include <hyprland/src/render/OpenGL.hpp>
#include <hyprland/src/render/Renderer.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
    HANDLE handle = nullptr;

    SP<Config::Values::CBoolValue> cfgEnabled;
    SP<Config::Values::CIntValue>  cfgCrossfadeMs;
    SP<Config::Values::CIntValue>  cfgNewBufferTimeoutMs;
    SP<Config::Values::CIntValue>  cfgHoldMs;
    SP<Config::Values::CBoolValue>   cfgReadiness;
    SP<Config::Values::CStringValue> cfgReadinessDump;
    SP<Config::Values::CStringValue> cfgPrepWorker;
    SP<Config::Values::CBoolValue>   cfgMesh;
    SP<Config::Values::CIntValue>    cfgMeshRampMs;

    struct STracked {
        PHLWINDOWREF               window;
        Vector2D                   baselineGoal;
        CResizeTransition*         transition = nullptr; // owned by window->m_transformers; valid while window is alive and attached
        WP<CWLSurfaceResource>     surface;
        CHyprSignalListener        precommit, commit;
        // Readiness bookkeeping: main-surface commits are what makes a thumbnail fresh (desynced subsurfaces are captured but not counted).
        uint64_t                   generation = 0, commits = 0, notedCommits = 0, sampledCommits = UINT64_MAX, aliasGeneration = 0;
        // Prep bookkeeping: selection counts this generation's readiness selections and revocations, so an answer for an earlier
        // selection is recognised as stale; stageWanted = a selection whose analysis readbacks found no free slot yet.
        uint64_t                   resizeNs = 0;
        uint32_t                   selection = 0;
        bool                       stageWanted = false;
    };
    // unique_ptr: precommit lambdas hold STracked*, which must survive vector growth.
    std::vector<std::unique_ptr<STracked>> tracked;

    CHyprSignalListener preChecksListener;

    // Analysis images for the prep worker: content downsampled by ANALYSIS_DIVISOR, the 1/4 device px the prep pipeline and its acceptance
    // gates were built at. 4 slots = old + new image of two windows in flight; further windows of one relayout retry on the next frame.
    constexpr int   ANALYSIS_DIVISOR = 4;
    Readback::CPool analysisPool{4, 0, 0};
    // A selection whose two readbacks are in flight; newTicket is the later one, so the pair is complete when it arrives.
    struct SStaged {
        uint64_t     window = 0, generation = 0, oldTicket = 0, newTicket = 0;
        uint32_t     selection = 0;
        Prep::SImage oldImage, newImage;
    };
    std::vector<SStaged> staged;

    // stale = answers for a superseded generation or selection; inHold / afterHold = usable answers by whether the hold was still on;
    // meshDrawn / meshLate = usable answers the transition took / refused because its box was already moving; meshRevoked = drawn
    // meshes whose selection was revoked.
    struct SPrepCounters {
        unsigned long staged = 0, stageRetries = 0, tooSmall = 0, stale = 0, inHold = 0, afterHold = 0, meshDrawn = 0, meshLate = 0, meshRevoked = 0;
    } prepCounts;
    std::vector<double> meshReadyMs; // resize -> usable answer, per answer

    // Compositor-thread cost of onPreChecks on frames where a transition was attached (capture + thumbnail submit/poll), ms. Logged at unload.
    std::vector<double> preChecksMs;

    uint64_t windowId(const PHLWINDOW& w) {
        return reinterpret_cast<uint64_t>(w.get());
    }

    uint64_t nowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Reaches Hyprland's logger only because the Makefile builds with -fno-gnu-unique (see there).
    void log(const std::string& msg) {
        Log::logger->log(Log::DEBUG, "[resize-transition] {}", msg);
    }

    // reason by value: callers pass t.transition->m_failure, which the erase below destroys before the log line is built.
    void detach(STracked& t, std::string reason) {
        if (!t.transition)
            return;
        if (const auto w = t.window.lock()) {
            std::erase_if(w->m_transformers, [&](const auto& tr) { return tr.get() == t.transition; });
            g_pHyprRenderer->damageWindow(w);
        }
        t.transition = nullptr;
        if (const auto w = t.window.lock()) {
            Hold::release(w, Hold::Clock::now());
            Readiness::endGeneration(windowId(w));
        }
        if (!reason.empty())
            log("native drawing: " + reason);
    }

    // D3-3: hold the size/position animation for up to hold_ms; a usable prep answer releases it earlier (onPrepAnswers).
    void hold(const PHLWINDOW& w) {
        if (cfgHoldMs->value() <= 0)
            return;
        std::string why;
        if (Hold::begin(w, Hold::Clock::now(), std::chrono::milliseconds{cfgHoldMs->value()}, why))
            log(std::format("hold {} ms: size {} -> {}", cfgHoldMs->value(), w->sizeAnimation()->value(), w->sizeAnimation()->goal()));
        else
            log("no hold: " + why);
    }

    // source: which hook noticed the goal change first ("precommit" = the client answered before any frame; capture-timing evidence).
    void checkGeneration(STracked& t, PHLWINDOW w, const char* source) {
        const auto goal = w->size(Desktop::View::IGeometric::GEOMETRIC_GOAL);
        if (goal == t.baselineGoal)
            return;
        log(std::format("generation {}x{} -> {}x{} via {}", t.baselineGoal.x, t.baselineGoal.y, goal.x, goal.y, source));
        t.baselineGoal = goal;
        if (!cfgEnabled->value() || !g_pHyprRenderer->shouldRenderWindow(w)) {
            detach(t, "");
            return;
        }
        if (t.transition) {
            if (!t.transition->beginGeneration(goal)) {
                detach(t, t.transition->m_failure);
                return;
            }
        } else {
            auto transition = makeUnique<CResizeTransition>(w);
            if (!transition->beginGeneration(goal)) {
                log("native drawing: " + transition->m_failure);
                return;
            }
            t.transition = transition.get();
            w->m_transformers.emplace_back(std::move(transition));
            g_pHyprRenderer->damageWindow(w);
        }
        hold(w);
        ++t.generation;
        t.sampledCommits = UINT64_MAX;
        t.resizeNs       = nowNs();
        t.selection      = 0;
        t.stageWanted    = false;
        auto dump        = cfgReadinessDump->value();
        Readiness::beginGeneration(windowId(w), t.generation, t.resizeNs, dump == "[[EMPTY]]" ? "" : dump);
    }

    void attachSurface(STracked& t, PHLWINDOW w) {
        const auto surface = w->resource();
        if (surface == t.surface.lock())
            return;
        t.surface   = surface;
        t.precommit = nullptr;
        t.commit    = nullptr;
        if (surface) {
            t.precommit = surface->m_events.precommit.listen([raw = &t] {
                if (const auto win = raw->window.lock(); win && win->m_isMapped)
                    checkGeneration(*raw, win, "precommit");
            });
            t.commit = surface->m_events.commit.listen([raw = &t] { ++raw->commits; });
        }
    }

    // Lazy attach: first sight takes the current goal as baseline, so a window that just mapped (no old content) never transitions.
    STracked& track(PHLWINDOW w) {
        for (auto& t : tracked)
            if (t->window.lock() == w)
                return *t;
        auto t          = std::make_unique<STracked>();
        t->window       = w;
        t->baselineGoal = w->size(Desktop::View::IGeometric::GEOMETRIC_GOAL);
        tracked.push_back(std::move(t));
        return *tracked.back();
    }

    void syncWindows() {
        // Expired refs never match a new window, even one allocated at the dead window's address.
        std::erase_if(tracked, [](const auto& t) { return t->window.expired(); });

        for (const auto& w : Desktop::windowState()->windows()) {
            if (!w->m_isMapped || w->isHidden())
                continue;
            auto& t = track(w);
            attachSurface(t, w);
            checkGeneration(t, w, "frame");
        }
    }

    // Readiness FrameSource: one thumbnail of this frame's new-content capture per fresh main-surface commit.
    void sample(STracked& t, const PHLWINDOW& w) {
        const auto* content = t.transition->newContent();
        if (!content || !cfgReadiness->value())
            return;
        const auto id = windowId(w);
        if (t.commits != t.notedCommits) {
            t.notedCommits = t.commits;
            Readiness::noteCommit(id, t.commits);
        }
        if (t.commits == t.sampledCommits)
            return;
        const auto& size = content->fb->m_size;
        const auto  tex  = content->fb->getTexture();
        if (!tex || size.x < 1 || size.y < 1)
            return;
        const auto newPx  = content->contentPx;
        const int  height = Thumbnail::heightFor(newPx.y / newPx.x);
        // D5-1 alias first, once per generation: the PBO pool completes in submission order, so the worker always has it before the first
        // frame. No frame goes out without it -- a frame observed alias-less would bypass the placeholder rule.
        if (t.aliasGeneration != t.generation) {
            const auto& old    = t.transition->oldContent();
            const auto  oldTex = old.fb ? old.fb->getTexture() : nullptr;
            const auto  monitor = w->m_monitor.lock();
            if (!oldTex || !monitor)
                return;
            const Vector2D shared{std::min(old.contentPx.x, newPx.x), std::min(old.contentPx.y, newPx.y)};
            // 10 logical px, as aliases_from() masks the recorded traces' crop alias.
            const int             inset = static_cast<int>(std::lround(10. * monitor->m_scale * Thumbnail::WIDTH / newPx.x));
            const Thumbnail::STag aliasTag{.window = id, .generation = t.generation, .sampleNs = nowNs(), .alias = true, .aliasInset = inset};
            if (!Thumbnail::submit(oldTex->m_texID, 0.F, 0.F, shared.x / old.fb->m_size.x, shared.y / old.fb->m_size.y,
                                   static_cast<int>(std::lround(Thumbnail::WIDTH * shared.x / newPx.x)),
                                   static_cast<int>(std::lround(height * shared.y / newPx.y)), aliasTag))
                return;
            t.aliasGeneration = t.generation;
        }
        const Thumbnail::STag tag{.window = id, .generation = t.generation, .commit = t.commits, .sampleNs = nowNs()};
        if (Thumbnail::submit(tex->m_texID, 0.F, 0.F, newPx.x / size.x, newPx.y / size.y, Thumbnail::WIDTH, height, tag))
            t.sampledCommits = t.commits;
        else if (*Thumbnail::failure())
            log(std::string{"readiness thumbnails disabled: "} + Thumbnail::failure());
    }

    STracked* findTracked(uint64_t id) {
        for (auto& t : tracked)
            if (const auto w = t->window.lock(); w && windowId(w) == id)
                return t.get();
        return nullptr;
    }

    // Starts the two analysis readbacks of the current selection. false -> no free readback slot: stageWanted stays set, next frame retries.
    bool stage(STracked& t, const PHLWINDOW& w) {
        t.stageWanted       = false;
        const auto* content = t.transition->newContent();
        const auto& old     = t.transition->oldContent();
        const auto  oldTex  = old.fb ? old.fb->getTexture() : nullptr;
        const auto  newTex  = content ? content->fb->getTexture() : nullptr;
        if (!oldTex || !newTex)
            return true;
        // Whole analysis pixels only: the box filter is exact at an integer ratio, at the cost of < ANALYSIS_DIVISOR device px at the
        // right and bottom edges. The worker refuses images under 2 px a side.
        const auto extent = [](const Vector2D& px) { return Vector2D{std::floor(px.x / ANALYSIS_DIVISOR), std::floor(px.y / ANALYSIS_DIVISOR)}; };
        const auto oldSize = extent(old.contentPx), newSize = extent(content->contentPx);
        if (std::min({oldSize.x, oldSize.y, newSize.x, newSize.y}) < 2) {
            ++prepCounts.tooSmall;
            return true;
        }
        const auto submit = [](const CResizeTransition::SSnapshot& snap, unsigned texture, const Vector2D& size) {
            return analysisPool.submit(texture, 0.F, 0.F, size.x * ANALYSIS_DIVISOR / snap.fb->m_size.x, size.y * ANALYSIS_DIVISOR / snap.fb->m_size.y,
                                       static_cast<int>(size.x), static_cast<int>(size.y));
        };
        const bool room      = analysisPool.reserve(static_cast<int>(std::max(oldSize.x, newSize.x)), static_cast<int>(std::max(oldSize.y, newSize.y)));
        const auto oldTicket = room ? submit(old, oldTex->m_texID, oldSize) : 0;
        const auto newTicket = oldTicket ? submit(*content, newTex->m_texID, newSize) : 0;
        if (!newTicket) { // a lone old readback completes into no staged entry and is dropped
            if (*analysisPool.failure())
                log(std::string{"prep images disabled: "} + analysisPool.failure());
            else {
                t.stageWanted = true;
                ++prepCounts.stageRetries;
            }
            return false;
        }
        staged.push_back({.window = windowId(w), .generation = t.generation, .oldTicket = oldTicket, .newTicket = newTicket, .selection = t.selection, .oldImage = {}, .newImage = {}});
        ++prepCounts.staged;
        return true;
    }

    // Scheduler: a selection asks the prep worker for a mesh; a revocation invalidates whatever was asked so far.
    void onReadinessEvents() {
        for (const auto& e : Readiness::drain()) {
            auto* const t       = findTracked(e.window);
            const bool  current = t && t->transition && t->generation == e.generation;
            switch (e.kind) {
                case Readiness::SEvent::SELECTED:
                    log(std::format("readiness 0x{:x} gen {}: ready at {:.1f} ms", e.window, e.generation, e.elapsedMs));
                    if (current) {
                        ++t->selection;
                        stage(*t, t->window.lock());
                    }
                    break;
                case Readiness::SEvent::VETOED: log(std::format("readiness 0x{:x} gen {}: selection vetoed at {:.1f} ms", e.window, e.generation, e.elapsedMs)); break;
                case Readiness::SEvent::REVOKED:
                    log(std::format("readiness 0x{:x} gen {}: revoked at {:.1f} ms", e.window, e.generation, e.elapsedMs));
                    if (current) {
                        ++t->selection;
                        t->stageWanted = false;
                        if (t->transition->revokeMesh())
                            ++prepCounts.meshRevoked;
                    }
                    break;
                case Readiness::SEvent::SUMMARY: log("readiness summary " + e.text); break;
                case Readiness::SEvent::ERROR: log("readiness error: " + e.text); break;
            }
        }
    }

    // Finished analysis readbacks -> one prep request per selection that is still the window's current one.
    void sendStaged() {
        analysisPool.poll([](uint64_t ticket, int width, int height, std::vector<uint8_t>&& rgba) {
            // Tickets complete in order: an entry still waiting for an earlier ticket lost that readback.
            std::erase_if(staged, [&](const SStaged& s) { return s.newTicket < ticket; });
            const auto s = std::ranges::find_if(staged, [&](const SStaged& s) { return s.oldTicket == ticket || s.newTicket == ticket; });
            if (s == staged.end())
                return;
            (s->oldTicket == ticket ? s->oldImage : s->newImage) = Prep::SImage{width, height, std::move(rgba)};
            if (s->newTicket != ticket)
                return;
            const auto* t = findTracked(s->window);
            if (!s->oldImage.rgba.empty() && t && t->transition && t->generation == s->generation && t->selection == s->selection)
                Prep::request({s->window, s->generation, s->selection, std::move(s->oldImage), std::move(s->newImage)});
            else
                ++prepCounts.stale;
            staged.erase(s);
        });
    }

    // D3-3: the hold ends on mesh ready. The mesh is offered first, while the box is still held: released, it would be too late.
    void onPrepAnswers() {
        if (const auto failure = Prep::takeFailure())
            log("prep worker down, plain scale from here: " + *failure);
        for (auto& a : Prep::drain()) {
            const auto* t = findTracked(a.window);
            if (!t || !t->transition || t->generation != a.generation || t->selection != a.selection) {
                ++prepCounts.stale;
                continue;
            }
            if (!a.ok) {
                log(std::format("prep 0x{:x} gen {} selection {}: refused: {}", a.window, a.generation, a.selection, a.refusal));
                continue;
            }
            const auto w      = t->window.lock();
            const auto now    = Hold::Clock::now();
            const bool held   = Hold::held(w, now);
            const auto readyMs = (nowNs() - t->resizeNs) / 1e6;
            const auto leaves  = a.leaves.size();
            const bool drawn   = cfgMesh->value() && t->transition->offerMesh({std::move(a.leaves),
                                                                               {static_cast<double>(a.oldWidth), static_cast<double>(a.oldHeight)},
                                                                               {static_cast<double>(a.newWidth), static_cast<double>(a.newHeight)}});
            Hold::release(w, now);
            ++(held ? prepCounts.inHold : prepCounts.afterHold);
            ++(drawn ? prepCounts.meshDrawn : prepCounts.meshLate);
            meshReadyMs.push_back(readyMs);
            log(std::format("prep 0x{:x} gen {} selection {}: mesh ready {:.1f} ms after resize ({} leaves, prep {:.1f} ms, round trip {:.1f} ms), {}, {}",
                            a.window, a.generation, a.selection, readyMs, leaves, a.prepMs, a.roundTripMs, held ? "hold released" : "hold already over",
                            drawn ? "mesh drawn" : "plain scale"));
        }
    }

    // Worker path: the config value, or the build's default when empty. A changed value restarts the worker.
    void syncPrepWorker() {
        auto path = cfgPrepWorker->value();
        if (path.empty() || path == "[[EMPTY]]")
            path = PREP_WORKER_DEFAULT;
        if (path == Prep::workerPath())
            return;
        std::string error;
        const auto  started = std::chrono::steady_clock::now();
        const bool  ok      = Prep::start(path, error);
        const auto  ms      = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        log(ok ? std::format("prep worker started in {:.1f} ms: {}", ms, path) : "no prep worker, plain scale only: " + error);
    }

    void onPreChecks(PHLMONITOR monitor) {
        const auto started = std::chrono::steady_clock::now();
        bool       active  = false;
        syncPrepWorker();
        syncWindows();

        const STransitionConfig config{.crossfadeMs = cfgCrossfadeMs->value(), .newBufferTimeoutMs = cfgNewBufferTimeoutMs->value(), .meshRampMs = cfgMeshRampMs->value()};
        // The Presenter draws with raw GL on every attached frame, also on those that capture nothing.
        if (const auto gl = g_pHyprRenderer->glBackend())
            gl->makeEGLCurrent();
        for (auto& t : tracked) {
            if (!t->transition)
                continue;
            const auto w = t->window.lock();
            if (!w->m_isMapped) {
                detach(*t, "window unmapped");
                continue;
            }
            if (w->m_monitor.lock() != monitor)
                continue;
            active = true;
            if (!t->transition->prepareFrame(config, Hold::held(w, Hold::Clock::now()))) {
                detach(*t, t->transition->m_failure.empty() ? "transition done" : t->transition->m_failure);
                continue;
            }
            sample(*t, w);
            if (t->stageWanted)
                stage(*t, w);
            // Hyprland stops damaging the window when the size animation ends, but the crossfade, the new-buffer timeout and the final
            // "done" check all run in prepareFrame: keep one frame queued per frame while attached.
            g_pEventLoopManager->doLater([ref = t->window] {
                if (const auto win = ref.lock())
                    g_pHyprRenderer->damageWindow(win);
            });
        }
        if (const auto gl = g_pHyprRenderer->glBackend())
            gl->makeEGLCurrent();
        Thumbnail::poll([](Thumbnail::SFrame&& frame) { Readiness::submit(std::move(frame)); });
        onReadinessEvents();
        sendStaged();
        onPrepAnswers();
        if (active)
            preChecksMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    }
}

APICALL EXPORT std::string PLUGIN_API_VERSION() {
    return HYPRLAND_API_VERSION;
}

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE h) {
    handle = h;
    if (std::string{__hyprland_api_get_hash()} != __hyprland_api_get_client_hash())
        throw std::runtime_error("resize-transition: compositor ABI mismatch, rebuild against the running Hyprland");

    cfgEnabled = makeShared<Config::Values::CBoolValue>("plugin:resize_transition:enabled", "Draw window size changes through the plugin (0 = native drawing)", true);
    // 100 ms: D3-1 starting value (M3), the short crossfade around the old/new switch; tunable per user taste.
    cfgCrossfadeMs = makeShared<Config::Values::CIntValue>("plugin:resize_transition:crossfade_ms",
                                                           "Old-to-new crossfade length in ms, starting when the client commits the new-size buffer", 100,
                                                           Config::Values::SIntValueOptions{.min = 0});
    // 300 ms: covers the Discord pilot's main reflow (max 190 ms) and later update (max 255 ms) after geometry settled (findings.md).
    cfgNewBufferTimeoutMs = makeShared<Config::Values::CIntValue>("plugin:resize_transition:new_buffer_timeout_ms",
                                                                  "After the size animation ends, ms to keep the scaled old snapshot while waiting for the new-size buffer",
                                                                  300, Config::Values::SIntValueOptions{.min = 0});
    // 200 ms: D3-3 cap chosen by the user; readiness 75-160 ms + prep ~92 ms often exceeds it. 0 = no hold.
    cfgHoldMs = makeShared<Config::Values::CIntValue>("plugin:resize_transition:hold_ms",
                                                      "Max ms a window's size/position animation is held at its start while the mesh is prepared (0 = never hold)", 200,
                                                      Config::Values::SIntValueOptions{.min = 0});
    cfgReadiness     = makeShared<Config::Values::CBoolValue>("plugin:resize_transition:readiness", "Sample new content and run the readiness detector (0 = off)", true);
    cfgReadinessDump = makeShared<Config::Values::CStringValue>("plugin:resize_transition:readiness_dump",
                                                                "Directory for readiness thumbnails + readiness_replay manifests per resize (empty = off; debugging only)", "");
    cfgPrepWorker    = makeShared<Config::Values::CStringValue>("plugin:resize_transition:prep_worker",
                                                                "Path of the prep-worker executable the plugin runs (empty = the path it was built with)", "");
    cfgMesh       = makeShared<Config::Values::CBoolValue>("plugin:resize_transition:mesh", "Draw the prepared mesh when it is ready before the window starts to move (0 = plain scale only)", true);
    // 80 ms: D3-5 starting value (M3) for the mesh weight ramp.
    cfgMeshRampMs = makeShared<Config::Values::CIntValue>("plugin:resize_transition:mesh_ramp_ms",
                                                          "Ms over which a mesh whose content changed under it relaxes back to plain scale", 80,
                                                          Config::Values::SIntValueOptions{.min = 0});
    for (const auto& value : {SP<Config::Values::IValue>(cfgEnabled), SP<Config::Values::IValue>(cfgCrossfadeMs), SP<Config::Values::IValue>(cfgNewBufferTimeoutMs),
                              SP<Config::Values::IValue>(cfgHoldMs), SP<Config::Values::IValue>(cfgReadiness), SP<Config::Values::IValue>(cfgReadinessDump),
                              SP<Config::Values::IValue>(cfgPrepWorker), SP<Config::Values::IValue>(cfgMesh), SP<Config::Values::IValue>(cfgMeshRampMs)})
        if (!HyprlandAPI::addConfigValueV2(handle, value))
            throw std::runtime_error(std::string{"resize-transition: cannot register "} + value->name());

    if (std::string error; !Hold::init(error))
        log("animation hold disabled: " + error);
    Readiness::start({});
    // At load, not at the first resize: the worker needs 0.5-2 s (context, kernels, warm-up) before its first answer is fast.
    syncPrepWorker();

    // An idle compositor renders no frame until the resize itself, so waiting for preChecks would take the new goal as baseline.
    syncWindows();
    preChecksListener = Event::bus()->m_events.render.preChecks.listen([](PHLMONITOR monitor) { onPreChecks(monitor); });
    log("loaded");
    return {"resize-transition", "Plain-scale resize transition with old snapshot and crossfade", "wavjaby", "0.1"};
}

APICALL EXPORT void PLUGIN_EXIT() {
    preChecksListener = nullptr;
    const auto& c = Hold::counters();
    log(std::format("hold counters: holds {} retargets {} released_early {} refused_spent {} refused_spring {} refused_idle {} adopted {} stale_dropped {}",
                    c.holds, c.retargets, c.releasedEarly, c.refusedSpent, c.refusedSpring, c.refusedIdle, c.adopted, c.staleDropped));
    Hold::exit();
    // Transformers' vtables live in this .so: every attached one must leave its window before unload.
    if (const auto gl = g_pHyprRenderer->glBackend())
        gl->makeEGLCurrent();
    for (auto& t : tracked)
        detach(*t, "");
    tracked.clear();
    // The worker runs this library's code: join before unmap. Summaries of the generations closed above are logged here.
    Readiness::stop();
    onReadinessEvents();
    Prep::stop();
    const auto pc = Prep::counters();
    log(std::format("prep counters: staged {} stage_retries {} too_small {} requests {} dropped {} replaced {} ok {} refused {} stale {} in_hold {} after_hold {}",
                    prepCounts.staged, prepCounts.stageRetries, prepCounts.tooSmall, pc.requests, pc.dropped, pc.replaced, pc.ok, pc.refused, prepCounts.stale,
                    prepCounts.inHold, prepCounts.afterHold));
    log(std::format("mesh counters: drawn {} late {} revoked {}", prepCounts.meshDrawn, prepCounts.meshLate, prepCounts.meshRevoked));
    Presenter::destroy();
    if (!meshReadyMs.empty()) {
        std::ranges::sort(meshReadyMs);
        log(std::format("mesh ready ms after resize over {} answers: min {:.1f} median {:.1f} max {:.1f}", meshReadyMs.size(), meshReadyMs.front(),
                        meshReadyMs[meshReadyMs.size() / 2], meshReadyMs.back()));
    }
    const auto& ac = analysisPool.counters();
    log(std::format("analysis readback counters: submitted {} busy {} completed {} gl_errors {}", ac.submitted, ac.busy, ac.completed, ac.glErrors));
    analysisPool.destroy();
    staged.clear();
    if (!preChecksMs.empty()) {
        std::ranges::sort(preChecksMs);
        const auto at = [](double q) { return preChecksMs[std::min(preChecksMs.size() - 1, size_t(q * preChecksMs.size()))]; };
        log(std::format("preChecks ms over {} transition frames: median {:.3f} p95 {:.3f} max {:.3f}", preChecksMs.size(), at(.5), at(.95), preChecksMs.back()));
    }
    const auto& tc = Thumbnail::counters();
    log(std::format("thumbnail counters: submitted {} busy {} completed {} gl_errors {}", tc.submitted, tc.busy, tc.completed, tc.glErrors));
    Thumbnail::destroy();
}
