// Resize transition plugin, skeleton (S1-A): every window size change shows the old snapshot, then the new buffer, plain-scaled into the
// animated box, with a short crossfade. Replaces the native path's UV stretch/crop (gap on shrink, progressive crop on grow, F12).
//
// Generation trigger: GEOMETRIC_GOAL size change, cause-agnostic (dispatcher, togglefloating, relayout, fullscreen).
// Capture timing: m_realSize's begin callback only queues sendWindowSize via doLater, and m_events.resize fires on animation ticks, which may
// run inside renderMonitor. So the goal is checked (and the old snapshot captured) at the two points that precede any texture replacement
// and run outside a render pass: the window surface's precommit, and render.preChecks of every monitor.
// Each generation also holds the size/position animation for up to hold_ms (Hold.hpp, D3-3); the old/new switch waits for the release.
// Readiness (S5): thumbnails of the new-content capture feed a per-generation detector on a worker thread; today it only logs decisions.
#include "Hold.hpp"
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

    struct STracked {
        PHLWINDOWREF               window;
        Vector2D                   baselineGoal;
        CResizeTransition*         transition = nullptr; // owned by window->m_transformers; valid while window is alive and attached
        WP<CWLSurfaceResource>     surface;
        CHyprSignalListener        precommit, commit;
        // Readiness bookkeeping: main-surface commits are what makes a thumbnail fresh (desynced subsurfaces are captured but not counted).
        uint64_t                   generation = 0, commits = 0, notedCommits = 0, sampledCommits = UINT64_MAX;
    };
    // unique_ptr: precommit lambdas hold STracked*, which must survive vector growth.
    std::vector<std::unique_ptr<STracked>> tracked;

    CHyprSignalListener preChecksListener;

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

    // D3-3: hold the size/position animation for up to hold_ms. Release today is the cap only; mesh-ready release comes with S5 prep.
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
        auto dump        = cfgReadinessDump->value();
        Readiness::beginGeneration(windowId(w), t.generation, nowNs(), dump == "[[EMPTY]]" ? "" : dump);
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
        const Thumbnail::STag tag{.window = id, .generation = t.generation, .commit = t.commits, .sampleNs = nowNs()};
        if (Thumbnail::submit(tex->m_texID, 0.F, 0.F, content->contentPx.x / size.x, content->contentPx.y / size.y,
                              content->contentPx.y / content->contentPx.x, tag))
            t.sampledCommits = t.commits;
        else if (*Thumbnail::failure())
            log(std::string{"readiness thumbnails disabled: "} + Thumbnail::failure());
    }

    void logReadiness() {
        for (const auto& e : Readiness::drain()) {
            switch (e.kind) {
                case Readiness::SEvent::SELECTED: log(std::format("readiness 0x{:x} gen {}: ready at {:.1f} ms", e.window, e.generation, e.elapsedMs)); break;
                case Readiness::SEvent::VETOED: log(std::format("readiness 0x{:x} gen {}: selection vetoed at {:.1f} ms", e.window, e.generation, e.elapsedMs)); break;
                case Readiness::SEvent::REVOKED: log(std::format("readiness 0x{:x} gen {}: revoked at {:.1f} ms", e.window, e.generation, e.elapsedMs)); break;
                case Readiness::SEvent::SUMMARY: log("readiness summary " + e.text); break;
                case Readiness::SEvent::ERROR: log("readiness error: " + e.text); break;
            }
        }
    }

    void onPreChecks(PHLMONITOR monitor) {
        const auto started = std::chrono::steady_clock::now();
        bool       active  = false;
        syncWindows();

        const STransitionConfig config{.crossfadeMs = cfgCrossfadeMs->value(), .newBufferTimeoutMs = cfgNewBufferTimeoutMs->value()};
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
        logReadiness();
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
    for (const auto& value : {SP<Config::Values::IValue>(cfgEnabled), SP<Config::Values::IValue>(cfgCrossfadeMs), SP<Config::Values::IValue>(cfgNewBufferTimeoutMs),
                              SP<Config::Values::IValue>(cfgHoldMs), SP<Config::Values::IValue>(cfgReadiness), SP<Config::Values::IValue>(cfgReadinessDump)})
        if (!HyprlandAPI::addConfigValueV2(handle, value))
            throw std::runtime_error(std::string{"resize-transition: cannot register "} + value->name());

    if (std::string error; !Hold::init(error))
        log("animation hold disabled: " + error);
    Readiness::start({});

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
    logReadiness();
    if (!preChecksMs.empty()) {
        std::ranges::sort(preChecksMs);
        const auto at = [](double q) { return preChecksMs[std::min(preChecksMs.size() - 1, size_t(q * preChecksMs.size()))]; };
        log(std::format("preChecks ms over {} transition frames: median {:.3f} p95 {:.3f} max {:.3f}", preChecksMs.size(), at(.5), at(.95), preChecksMs.back()));
    }
    const auto& tc = Thumbnail::counters();
    log(std::format("thumbnail counters: submitted {} busy {} completed {} gl_errors {}", tc.submitted, tc.busy, tc.completed, tc.glErrors));
    Thumbnail::destroy();
}
