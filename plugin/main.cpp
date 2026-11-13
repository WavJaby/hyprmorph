// Resize transition plugin: every window size change shows the old snapshot, then the new buffer, plain-scaled into the
// animated box, with a short crossfade. Replaces the native path's UV stretch/crop (gap on shrink, progressive crop on grow).
//
// Generation trigger: GEOMETRIC_GOAL size change, cause-agnostic (dispatcher, togglefloating, relayout, fullscreen).
// Capture timing: m_realSize's begin callback only queues sendWindowSize via doLater, and m_events.resize fires on animation ticks, which may
// run inside renderMonitor. So the goal is checked (and the old snapshot captured) at the two points that precede any texture replacement
// and run outside a render pass: the window surface's precommit, and render.preChecks of every monitor.
// Each generation also holds the size/position animation for up to hold_ms (Hold.hpp); the old/new switch waits for the release.
// Readiness: thumbnails of the new-content capture feed a per-generation detector on a worker thread.
// Prep: each readiness candidate -- a frame that opened a confirm window -- sends the old snapshot and the current new content, at
// analysis resolution, to the out-of-process prep worker, so the mesh is prepared while readiness confirms. Once the candidate is
// selected and its answer is in, the hold is released and, if the box had not moved yet, the answer becomes the mesh the Presenter draws.
#include "Hold.hpp"
#include "Prep.hpp"
#include "Presenter.hpp"
#include "Readback.hpp"
#include "Readiness.hpp"
#include "Thumbnail.hpp"
#include "Transition.hpp"
#include "../prep/src/json.hpp"

#include <hyprland/src/plugins/PluginAPI.hpp>
#include <hyprland/src/config/values/ConfigValues.hpp>
#include <hyprland/src/debug/log/Logger.hpp>
#include <hyprland/src/desktop/state/WindowState.hpp>
#include <hyprland/src/desktop/view/window/Window.hpp>
#include <hyprland/src/desktop/view/window/WindowEffectsController.hpp>
#include <hyprland/src/render/transformer/TransformerList.hpp>
#include <hyprland/src/event/EventBus.hpp>
#include <hyprland/src/managers/eventLoop/EventLoopManager.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/protocols/core/Compositor.hpp>
#include <hyprland/src/render/OpenGL.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/Compositor.hpp>

#include <xf86drm.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {
    HANDLE handle = nullptr;

    SP<Config::Values::CBoolValue> cfgEnabled;
    SP<Config::Values::CFloatValue> cfgCrossfadeSpan;
    SP<Config::Values::CIntValue>  cfgCrossfadeMs;
    SP<Config::Values::CFloatValue> cfgCrossfadeBlurPx;
    SP<Config::Values::CIntValue>  cfgNewBufferTimeoutMs;
    SP<Config::Values::CIntValue>  cfgHoldMs;
    SP<Config::Values::CBoolValue>   cfgReadiness;
    SP<Config::Values::CStringValue> cfgReadinessDump;
    SP<Config::Values::CStringValue> cfgPrepWorker;
    SP<Config::Values::CBoolValue>   cfgMesh;
    SP<Config::Values::CIntValue>    cfgMeshRampMs;
    SP<Config::Values::CBoolValue>   cfgVerbose;
    SP<IPC::Socket1::SCommand>       statsCommand;

    struct STracked {
        PHLWINDOWREF               window;
        Vector2D                   baselineGoal;
        CResizeTransition*         transition = nullptr; // owned by the window's transformer list; valid while window is alive and attached
        WP<CWLSurfaceResource>     surface;
        CHyprSignalListener        precommit, commit;
        // Readiness bookkeeping: main-surface commits are what makes a thumbnail fresh (desynced subsurfaces are captured but not counted).
        uint64_t                   generation = 0, commits = 0, notedCommits = 0, sampledCommits = UINT64_MAX, aliasGeneration = 0;
        // Prep bookkeeping, all per generation; candidate ids are the readiness detector's, 0 = none.
        // candidate = newest one seen; asked = the one whose images or request are out (one at a time per window: a newer candidate waits
        // for it, then is staged with the content of that moment); selected = the one readiness confirmed, cleared by a revocation;
        // answer = the worker's result for `candidate`, kept until that candidate is selected; stageWanted = `candidate` still needs staging;
        // staged = `candidate` has been staged (a selection that finds it unstaged asks then).
        uint64_t                   resizeNs = 0;
        uint32_t                   candidate = 0, asked = 0, selected = 0;
        bool                       candidateStaged = false;
        // Group hold: windows whose generations began in one layout pass form a group and leave the hold together. settled = this window is
        // done waiting: its mesh is in, or none will come (refusal, no worker, images too small).
        uint64_t                   group = 0;
        bool                       settled = false, waitCounted = false;
        std::optional<Prep::SAnswer> answer;
        bool                       stageWanted = false;
    };
    // unique_ptr: precommit lambdas hold STracked*, which must survive vector growth.
    std::vector<std::unique_ptr<STracked>> tracked;

    CHyprSignalListener preChecksListener;

    // Analysis images for the prep worker: content downsampled by ANALYSIS_DIVISOR, the 1/4 device px the prep pipeline and its acceptance
    // gates were built at. 4 slots = old + new image of two windows in flight; further windows of one relayout retry on the next frame.
    constexpr int   ANALYSIS_DIVISOR = 4;
    Readback::CPool analysisPool{4, 0, 0};
    // A candidate whose two readbacks are in flight; newTicket is the later one, so the pair is complete when it arrives.
    struct SStaged {
        uint64_t     window = 0, generation = 0, oldTicket = 0, newTicket = 0;
        uint32_t     candidate = 0;
        Prep::SImage oldImage, newImage;
    };
    std::vector<SStaged> staged;

    // stale = answers for a superseded generation or candidate (speculation that did not pay); early = usable answers that were in before
    // their candidate was selected; inHold / afterHold = usable answers by whether the hold was still on; meshDrawn / meshLate = usable
    // answers the transition took / refused because its box was past the crossfade window; meshJoined = drawn meshes taken after the box
    // started moving; meshRevoked = drawn meshes whose selection was revoked.
    struct SPrepCounters {
        unsigned long staged = 0, stageRetries = 0, tooSmall = 0, stale = 0, early = 0, inHold = 0, afterHold = 0, meshDrawn = 0, meshLate = 0, meshJoined = 0,
                      meshRevoked = 0;
    } prepCounts;
    std::vector<double> meshReadyMs; // resize -> usable answer, per answer

    // Newest usable or refused answers, for `hyprctl hyprmorph`: what each resize got, without a log.
    struct SAnswerRecord {
        uint64_t    window = 0, generation = 0;
        uint32_t    candidate = 0;
        double      readyMs = 0, prepMs = 0;
        double      resizeUnix = 0; // wall-clock s of the resize (now - readyMs): matches answers to a screen recording's own clock
        std::string outcome;        // drawn / joined / late / refused
    };
    constexpr size_t          ANSWER_HISTORY = 64;
    std::deque<SAnswerRecord> answers;

    // Why transitions ended (detach reasons, "transition done" included), readiness events, and the prep worker's life.
    std::map<std::string, unsigned long> detachReasons;
    struct SReadinessCounters {
        unsigned long selected = 0, vetoed = 0, revoked = 0, errors = 0;
    } readinessCounts;
    struct SWorkerState {
        unsigned long downs = 0, spawnFailures = 0;
        std::string   lastError;
    } workerState;

    // Hold group being filled: every generation that begins before the end of the next syncWindows pass joins it -- one layout
    // recalculation sets all its goals before the next render.preChecks, whether a frame or a client's precommit notices them first.
    uint64_t openGroup = 1;
    bool     openGroupUsed = false;
    // groupReleases = groups of several windows released together; groupWaits = windows that were settled but held for a slower member.
    struct SGroupCounters {
        unsigned long groupReleases = 0, groupWaits = 0;
    } groupCounts;

    // Compositor-thread cost of onPreChecks on frames where a transition was attached (capture + thumbnail submit/poll), ms.
    std::vector<double> preChecksMs;

    struct STracked;
    PHLWINDOW live(STracked& t);

    uint64_t windowId(const PHLWINDOW& w) {
        return reinterpret_cast<uint64_t>(w.get());
    }

    uint64_t nowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Both reach Hyprland's logger only because the Makefile builds with -fno-gnu-unique (see there).
    // warn: a part of the plugin stopped working (always written). trace: per-resize events, only with plugin:hyprmorph:verbose.
    // Counters are not logged at all: `hyprctl hyprmorph` (statsJson) returns them on demand, whatever the log settings.
    void warn(const std::string& msg) {
        Log::logger->log(Log::WARN, "[hyprmorph] {}", msg);
    }
    void trace(const std::string& msg) {
        if (cfgVerbose && cfgVerbose->value())
            Log::logger->log(Log::DEBUG, "[hyprmorph] {}", msg);
    }

    void recordAnswer(SAnswerRecord record) {
        record.resizeUnix = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() - record.readyMs / 1000.;
        answers.push_back(std::move(record));
        if (answers.size() > ANSWER_HISTORY)
            answers.pop_front();
    }

    // reason by value: callers pass t.transition->m_failure, which the erase below destroys before the log line is built.
    void detach(STracked& t, std::string reason) {
        if (!t.transition)
            return;
        if (const auto w = t.window.lock()) {
            t.transition->deactivate();
            w->effects().transformers()->removeInactive();
            g_pHyprRenderer->damageWindow(w);
        }
        t.transition = nullptr;
        if (const auto w = t.window.lock()) {
            Hold::release(w, Hold::Clock::now());
            Readiness::endGeneration(windowId(w));
        }
        if (!reason.empty()) {
            ++detachReasons[reason];
            trace("native drawing: " + reason);
        }
    }

    // Hold the size/position animation for up to hold_ms; a usable prep answer releases it earlier (onPrepAnswers).
    void hold(const PHLWINDOW& w) {
        if (cfgHoldMs->value() <= 0)
            return;
        std::string why;
        if (Hold::begin(w, Hold::Clock::now(), std::chrono::milliseconds{cfgHoldMs->value()}, why))
            trace(std::format("hold {} ms: size {} -> {}", cfgHoldMs->value(), w->sizeAnimation()->value(), w->sizeAnimation()->goal()));
        else
            trace("no hold: " + why);
    }

    // source: which hook noticed the goal change first ("precommit" = the client answered before any frame; capture-timing evidence).
    void checkGeneration(STracked& t, PHLWINDOW w, const char* source) {
        const auto goal = w->size(Desktop::View::IGeometric::GEOMETRIC_GOAL);
        if (goal == t.baselineGoal)
            return;
        trace(std::format("generation {}x{} -> {}x{} via {}", t.baselineGoal.x, t.baselineGoal.y, goal.x, goal.y, source));
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
            // The list constructs its transformers itself: a generation that cannot begin leaves it again at once.
            auto* const transition = w->effects().transformers()->emplace<CResizeTransition>(PHLWINDOWREF{w});
            if (!transition->beginGeneration(goal)) {
                const auto reason = transition->m_failure;
                transition->deactivate();
                w->effects().transformers()->removeInactive();
                ++detachReasons[reason];
                trace("native drawing: " + reason);
                return;
            }
            t.transition = transition;
            g_pHyprRenderer->damageWindow(w);
        }
        hold(w);
        ++t.generation;
        t.sampledCommits = UINT64_MAX;
        t.resizeNs       = nowNs();
        t.candidate = t.asked = t.selected = 0;
        t.candidateStaged = false;
        t.answer.reset();
        t.group       = openGroup;
        openGroupUsed = true;
        t.settled     = !cfgReadiness->value() || !Prep::running();
        t.waitCounted = false;
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
                if (const auto win = raw->window.lock(); win && win->mapped())
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
            if (!w->mapped() || w->isHidden())
                continue;
            auto& t = track(w);
            attachSurface(t, w);
            checkGeneration(t, w, "frame");
        }
        if (openGroupUsed) {
            ++openGroup;
            openGroupUsed = false;
        }
    }

    // Readiness frame source: one thumbnail of this frame's new-content capture per fresh main-surface commit.
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
        // Old-content placeholder alias first, once per generation: the PBO pool completes in submission order, so the worker always has it before the first
        // frame. No frame goes out without it -- a frame observed alias-less would bypass the placeholder rule.
        if (t.aliasGeneration != t.generation) {
            const auto& old    = t.transition->oldContent();
            const auto  oldTex = old.fb ? old.fb->getTexture() : nullptr;
            const auto  monitor = w->m_monitor.lock();
            if (!oldTex || !monitor)
                return;
            const Vector2D shared{std::min(old.contentPx.x, newPx.x), std::min(old.contentPx.y, newPx.y)};
            // 10 logical px: the mask inset the placeholder threshold was derived with on recorded resizes.
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
            warn(std::string{"readiness thumbnails disabled: "} + Thumbnail::failure());
    }

    // The tracked window, or null once it is gone. A window can be destroyed in the middle of a pass -- rendering a snapshot of another
    // window lets Hyprland finish a closing one's fade-out -- and its transformer dies with it: the pointer must not be used again.
    PHLWINDOW live(STracked& t) {
        auto w = t.window.lock();
        if (!w)
            t.transition = nullptr;
        return w;
    }

    STracked* findTracked(uint64_t id) {
        for (auto& t : tracked)
            if (const auto w = live(*t); w && windowId(w) == id)
                return t.get();
        return nullptr;
    }

    // Starts the two analysis readbacks for the window's newest candidate, unless an earlier one is still out. A readback slot that is
    // not free leaves stageWanted set: the next frame retries.
    void stage(STracked& t, const PHLWINDOW& w) {
        t.stageWanted = true;
        if (t.asked)
            return;
        const auto* content = t.transition->newContent();
        const auto& old     = t.transition->oldContent();
        const auto  oldTex  = old.fb ? old.fb->getTexture() : nullptr;
        const auto  newTex  = content ? content->fb->getTexture() : nullptr;
        if (!oldTex || !newTex)
            return;
        // Whole analysis pixels only: the box filter is exact at an integer ratio, at the cost of < ANALYSIS_DIVISOR device px at the
        // right and bottom edges. The worker refuses images under 2 px a side.
        const auto extent = [](const Vector2D& px) { return Vector2D{std::floor(px.x / ANALYSIS_DIVISOR), std::floor(px.y / ANALYSIS_DIVISOR)}; };
        const auto oldSize = extent(old.contentPx), newSize = extent(content->contentPx);
        if (std::min({oldSize.x, oldSize.y, newSize.x, newSize.y}) < 2) {
            ++prepCounts.tooSmall;
            t.stageWanted = false;
            t.settled     = true;
            return;
        }
        const auto submit = [](const CResizeTransition::SSnapshot& snap, unsigned texture, const Vector2D& size) {
            return analysisPool.submit(texture, 0.F, 0.F, size.x * ANALYSIS_DIVISOR / snap.fb->m_size.x, size.y * ANALYSIS_DIVISOR / snap.fb->m_size.y,
                                       static_cast<int>(size.x), static_cast<int>(size.y));
        };
        const bool room      = analysisPool.reserve(static_cast<int>(std::max(oldSize.x, newSize.x)), static_cast<int>(std::max(oldSize.y, newSize.y)));
        const auto oldTicket = room ? submit(old, oldTex->m_texID, oldSize) : 0;
        const auto newTicket = oldTicket ? submit(*content, newTex->m_texID, newSize) : 0;
        if (!newTicket) { // a lone old readback completes into no staged entry and is dropped
            if (*analysisPool.failure()) {
                warn(std::string{"prep images disabled: "} + analysisPool.failure());
                t.stageWanted = false;
            } else
                ++prepCounts.stageRetries;
            return;
        }
        staged.push_back({.window = windowId(w), .generation = t.generation, .oldTicket = oldTicket, .newTicket = newTicket, .candidate = t.candidate, .oldImage = {}, .newImage = {}});
        t.asked           = t.candidate;
        t.candidateStaged = true;
        t.stageWanted     = false;
        ++prepCounts.staged;
    }

    STransitionConfig transitionConfig() {
        return {.crossfadeSpan      = cfgCrossfadeSpan->value(),
                .crossfadeMs        = cfgCrossfadeMs->value(),
                .newBufferTimeoutMs = cfgNewBufferTimeoutMs->value(),
                .meshRampMs         = cfgMeshRampMs->value(),
                .crossfadeBlurPx    = cfgCrossfadeBlurPx->value()};
    }

    // Mesh ready = the selected candidate's answer is in. Offered here: during the hold it plays from the box's start, after it the
    // transition joins the moving box if that is still short of the crossfade window. The hold itself ends in releaseGroups, once the whole
    // group is settled.
    void useAnswer(STracked& t) {
        auto       a = std::move(*t.answer);
        t.answer.reset();
        const auto w       = t.window.lock();
        const auto now     = Hold::Clock::now();
        const bool held    = Hold::held(w, now);
        const auto readyMs = (nowNs() - t.resizeNs) / 1e6;
        const auto leaves  = a.leaves.size();
        bool       joined  = false;
        const bool drawn   = cfgMesh->value() && t.transition->offerMesh({std::move(a.leaves),
                                                                          {static_cast<double>(a.oldWidth), static_cast<double>(a.oldHeight)},
                                                                          {static_cast<double>(a.newWidth), static_cast<double>(a.newHeight)}},
                                                                         joined);
        t.settled = true;
        ++(held ? prepCounts.inHold : prepCounts.afterHold);
        ++(drawn ? prepCounts.meshDrawn : prepCounts.meshLate);
        if (joined)
            ++prepCounts.meshJoined;
        meshReadyMs.push_back(readyMs);
        recordAnswer({.window = a.window, .generation = a.generation, .candidate = a.candidate, .readyMs = readyMs, .prepMs = a.prepMs,
                      .outcome = !drawn ? "late" : joined ? "joined" : "drawn"});
        trace(std::format("prep 0x{:x} gen {} candidate {}: mesh ready {:.1f} ms after resize ({} leaves, prep {:.1f} ms, round trip {:.1f} ms), {}, {}", a.window,
                        a.generation, a.candidate, readyMs, leaves, a.prepMs, a.roundTripMs, held ? "hold released" : "hold already over",
                        !drawn ? "plain scale" : joined ? "mesh joined the moving box" : "mesh drawn"));
    }

    // A group leaves the hold when every member is settled; until then a settled member waits (its cap still bounds the wait).
    // A member whose transition ended, or that was retargeted into a newer group, no longer counts.
    void releaseGroups() {
        const auto now = Hold::Clock::now();
        for (auto& first : tracked) {
            const auto firstWindow = live(*first);
            if (!first->transition || !Hold::held(firstWindow, now))
                continue;
            size_t members = 0, waiting = 0;
            for (auto& t : tracked)
                if (live(*t) && t->transition && t->group == first->group) {
                    ++members;
                    waiting += !t->settled;
                }
            if (waiting) {
                if (first->settled && !first->waitCounted) {
                    first->waitCounted = true;
                    ++groupCounts.groupWaits;
                }
                continue;
            }
            for (auto& t : tracked)
                if (const auto w = live(*t); w && t->transition && t->group == first->group)
                    Hold::release(w, now);
            if (members > 1) {
                ++groupCounts.groupReleases;
                trace(std::format("group {}: {} windows released together {:.1f} ms after resize", first->group, members, (nowNs() - first->resizeNs) / 1e6));
            }
        }
    }

    // A new candidate supersedes the one before: its answer is for a layout the client has left.
    void wantCandidate(STracked& t, uint32_t candidate) {
        if (candidate != t.candidate) {
            t.candidate       = candidate;
            t.candidateStaged = false;
            t.answer.reset();
        }
        if (!t.candidateStaged)
            stage(t, t.window.lock());
    }

    // Scheduler: a candidate asks the prep worker for a mesh, its selection lets the answer through, a revocation takes the mesh back.
    void onReadinessEvents() {
        for (const auto& e : Readiness::drain()) {
            auto* const t       = findTracked(e.window);
            const bool  current = t && t->transition && t->generation == e.generation;
            switch (e.kind) {
                case Readiness::SEvent::CANDIDATE:
                    if (current)
                        wantCandidate(*t, e.candidate);
                    break;
                case Readiness::SEvent::SELECTED:
                    ++readinessCounts.selected;
                    trace(std::format("readiness 0x{:x} gen {}: ready at {:.1f} ms", e.window, e.generation, e.elapsedMs));
                    if (current) {
                        wantCandidate(*t, e.candidate); // a selection whose candidate was never announced (static layout) is asked now
                        t->selected = e.candidate;
                        if (t->answer) {
                            ++prepCounts.early;
                            useAnswer(*t);
                        }
                    }
                    break;
                case Readiness::SEvent::VETOED:
                    ++readinessCounts.vetoed;
                    trace(std::format("readiness 0x{:x} gen {}: selection vetoed at {:.1f} ms", e.window, e.generation, e.elapsedMs));
                    break;
                case Readiness::SEvent::REVOKED:
                    ++readinessCounts.revoked;
                    trace(std::format("readiness 0x{:x} gen {}: revoked at {:.1f} ms", e.window, e.generation, e.elapsedMs));
                    if (current) {
                        t->selected = 0;
                        if (t->transition->revokeMesh())
                            ++prepCounts.meshRevoked;
                    }
                    break;
                case Readiness::SEvent::SUMMARY: trace("readiness summary " + e.text); break;
                case Readiness::SEvent::ERROR:
                    ++readinessCounts.errors;
                    warn("readiness error: " + e.text);
                    break;
            }
        }
    }

    // The candidate that was out is settled (answered, superseded, or its images were lost): the newest one may go next.
    void settle(STracked& t, uint32_t candidate) {
        if (t.asked != candidate)
            return;
        t.asked = 0;
        if (t.candidate != candidate)
            t.stageWanted = true;
    }

    // Finished analysis readbacks -> the prep request of the candidate they were staged for.
    void sendStaged() {
        analysisPool.poll([](uint64_t ticket, int width, int height, std::vector<uint8_t>&& rgba) {
            // Tickets complete in order: an entry still waiting for an earlier ticket lost that readback.
            std::erase_if(staged, [&](const SStaged& s) {
                if (s.newTicket >= ticket)
                    return false;
                if (auto* const t = findTracked(s.window); t && t->generation == s.generation) {
                    settle(*t, s.candidate);
                    t->stageWanted = true;
                }
                return true;
            });
            const auto s = std::ranges::find_if(staged, [&](const SStaged& s) { return s.oldTicket == ticket || s.newTicket == ticket; });
            if (s == staged.end())
                return;
            (s->oldTicket == ticket ? s->oldImage : s->newImage) = Prep::SImage{width, height, std::move(rgba)};
            if (s->newTicket != ticket)
                return;
            // Sent even when a newer candidate has appeared meanwhile: its answer is what frees the window for the next one.
            if (const auto* t = findTracked(s->window); !s->oldImage.rgba.empty() && t && t->transition && t->generation == s->generation)
                Prep::request({s->window, s->generation, s->candidate, std::move(s->oldImage), std::move(s->newImage)});
            else
                ++prepCounts.stale;
            staged.erase(s);
        });
    }

    void onPrepAnswers() {
        if (const auto failure = Prep::takeFailure()) {
            ++workerState.downs;
            workerState.lastError = *failure;
            warn("prep worker down, plain scale from here: " + *failure);
            for (auto& t : tracked)
                t->settled = true;
        }
        for (auto& a : Prep::drain()) {
            auto* const t = findTracked(a.window);
            if (!t || !t->transition || t->generation != a.generation) {
                ++prepCounts.stale;
                continue;
            }
            settle(*t, a.candidate);
            if (t->candidate != a.candidate) {
                ++prepCounts.stale;
                continue;
            }
            if (!a.ok) {
                recordAnswer({.window = a.window, .generation = a.generation, .candidate = a.candidate, .readyMs = (nowNs() - t->resizeNs) / 1e6,
                              .prepMs = a.prepMs, .outcome = "refused"});
                trace(std::format("prep 0x{:x} gen {} candidate {}: refused: {}", a.window, a.generation, a.candidate, a.refusal));
                t->settled = true;
                continue;
            }
            t->answer = std::move(a);
            if (t->selected == t->candidate)
                useAnswer(*t);
        }
    }

    // Render node of the GPU the compositor renders on: the worker's analysis must run there, not on whichever GPU enumerates first.
    // Hyprland's render-node fd when it opened one, else the render node of its primary DRM device. Empty -> no node found.
    std::string compositorRenderNode() {
        for (const int fd : {g_pCompositor->m_drmRenderNode.fd, g_pCompositor->m_drm.fd}) {
            if (fd < 0)
                continue;
            if (char* name = drmGetRenderDeviceNameFromFd(fd)) {
                std::string node{name};
                free(name);
                return node;
            }
        }
        return {};
    }

    // Worker path: the config value, or the worker linked into this plugin when empty. A changed value restarts the worker.
    void syncPrepWorker() {
        auto path = cfgPrepWorker->value();
        if (path.empty() || path == "[[EMPTY]]")
            path = Prep::EMBEDDED;
        if (path == Prep::workerPath())
            return;
        std::string error;
        const auto  node    = compositorRenderNode();
        const auto  started = std::chrono::steady_clock::now();
        const bool  ok      = node.empty() ? (error = "compositor has no DRM render node", false) : Prep::start(path, node, error);
        const auto  ms      = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        if (ok)
            trace(std::format("prep worker started in {:.1f} ms: {} on {}", ms, path, node));
        else {
            ++workerState.spawnFailures;
            workerState.lastError = error;
            warn("no prep worker, plain scale only: " + error);
        }
    }

    void onPreChecks(PHLMONITOR monitor) {
        const auto started = std::chrono::steady_clock::now();
        bool       active  = false;
        syncPrepWorker();
        syncWindows();

        const STransitionConfig config = transitionConfig();
        // The Presenter draws with raw GL on every attached frame, also on those that capture nothing.
        if (const auto gl = g_pHyprRenderer->glBackend())
            gl->makeEGLCurrent();
        for (auto& t : tracked) {
            const auto w = live(*t);
            if (!t->transition)
                continue;
            if (!w->mapped()) {
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
            if (t->stageWanted && !t->asked)
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
        releaseGroups();
        if (active)
            preChecksMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    }

    std::string holdDisabled; // why the animation hold is off for this load, empty when on

    // Quantiles over an unsorted sample (copied): {n, min, median, p95, max}.
    prep::json::Value summary(std::vector<double> sample) {
        auto out = prep::json::Value::object();
        out.set("n", sample.size());
        if (sample.empty())
            return out;
        std::ranges::sort(sample);
        const auto at = [&](double q) { return sample[std::min(sample.size() - 1, size_t(q * sample.size()))]; };
        out.set("min", sample.front());
        out.set("median", at(.5));
        out.set("p95", at(.95));
        out.set("max", sample.back());
        return out;
    }

    // `hyprctl hyprmorph`: every counter of this load as one JSON object (the same whatever the output format flag).
    std::string statsJson() {
        const auto object = [](std::initializer_list<std::pair<const char*, prep::json::Value>> fields) {
            auto out = prep::json::Value::object();
            for (const auto& [key, value] : fields)
                out.set(key, value);
            return out;
        };
        const auto readback = [&](const Readback::SCounters& c) {
            return object({{"submitted", c.submitted}, {"busy", c.busy}, {"completed", c.completed}, {"gl_errors", c.glErrors}});
        };
        const auto& h  = Hold::counters();
        const auto  pc = Prep::counters();
        auto        reasons = prep::json::Value::object();
        for (const auto& [reason, count] : detachReasons)
            reasons.set(reason, count);
        auto recent = prep::json::Value::array();
        for (const auto& a : answers)
            recent.push(object({{"window", std::format("0x{:x}", a.window)}, {"generation", a.generation}, {"candidate", size_t(a.candidate)},
                                {"ready_ms", a.readyMs}, {"prep_ms", a.prepMs}, {"resize_unix", a.resizeUnix},
                                {"outcome", a.outcome}}));
        const auto root = object({
            {"worker", object({{"running", Prep::running()}, {"path", Prep::workerPath()}, {"node", Prep::renderNode()}, {"downs", workerState.downs},
                               {"spawn_failures", workerState.spawnFailures}, {"last_error", workerState.lastError}})},
            {"hold", object({{"disabled", holdDisabled}, {"holds", h.holds}, {"retargets", h.retargets}, {"released_early", h.releasedEarly},
                             {"refused_spent", h.refusedSpent}, {"refused_spring", h.refusedSpring}, {"refused_idle", h.refusedIdle}, {"adopted", h.adopted},
                             {"stale_dropped", h.staleDropped}})},
            {"readiness", object({{"selected", readinessCounts.selected}, {"vetoed", readinessCounts.vetoed}, {"revoked", readinessCounts.revoked},
                                  {"errors", readinessCounts.errors}})},
            {"prep", object({{"staged", prepCounts.staged}, {"stage_retries", prepCounts.stageRetries}, {"too_small", prepCounts.tooSmall},
                             {"requests", pc.requests}, {"dropped", pc.dropped}, {"replaced", pc.replaced}, {"ok", pc.ok}, {"refused", pc.refused},
                             {"stale", prepCounts.stale}, {"early", prepCounts.early}, {"in_hold", prepCounts.inHold}, {"after_hold", prepCounts.afterHold}})},
            {"mesh", object({{"drawn", prepCounts.meshDrawn}, {"joined", prepCounts.meshJoined}, {"late", prepCounts.meshLate}, {"revoked", prepCounts.meshRevoked}})},
            {"group", object({{"group_releases", groupCounts.groupReleases}, {"group_waits", groupCounts.groupWaits}})},
            {"readback", object({{"analysis", readback(analysisPool.counters())}, {"thumbnail", readback(Thumbnail::counters())}})},
            {"detach", reasons},
            {"mesh_ready_ms", summary(meshReadyMs)},
            {"prechecks_ms", summary(preChecksMs)},
            {"answers", recent},
        });
        return prep::json::dump(root);
    }
}

APICALL EXPORT std::string PLUGIN_API_VERSION() {
    return HYPRLAND_API_VERSION;
}

APICALL EXPORT PLUGIN_DESCRIPTION_INFO PLUGIN_INIT(HANDLE h) {
    handle = h;
    if (std::string{__hyprland_api_get_hash()} != __hyprland_api_get_client_hash())
        throw std::runtime_error("hyprmorph: compositor ABI mismatch, rebuild against the running Hyprland");

    cfgEnabled = makeShared<Config::Values::CBoolValue>("plugin:hyprmorph:enabled", "Draw window size changes through the plugin (0 = native drawing)", true);
    // 0.3: t 0.5..0.8. Without a mesh that is box progress, ~62..128 ms after release on a 479 ms easeOutQuint animation,
    // box speed 73 % -> 32 % of its peak. Starting at the switch, not before it, lets a late mesh join until box progress 0.5. Tunable.
    cfgCrossfadeSpan = makeShared<Config::Values::CFloatValue>("plugin:hyprmorph:crossfade_span",
                                                               "Old-to-new crossfade length in transition progress (0..1), starting at the old/new switch at 0.5", .3F,
                                                               Config::Values::SFloatValueOptions{.min = 0.F, .max = 1.F});
    // 1.5 px: untuned starting value; in a synthetic study 12 px made text indistinct.
    cfgCrossfadeBlurPx = makeShared<Config::Values::CFloatValue>("plugin:hyprmorph:crossfade_blur_px",
                                                                 "Peak blur (Gaussian sigma, logical px) of the content in the middle of the old/new crossfade (0 = off)",
                                                                 1.5F, Config::Values::SFloatValueOptions{.min = 0.F, .max = 4.F});
    // 100 ms: untuned starting value; only for new content that arrives after the crossfade window opened.
    cfgCrossfadeMs = makeShared<Config::Values::CIntValue>("plugin:hyprmorph:crossfade_ms",
                                                           "Old-to-new crossfade length in ms when the new-size buffer arrives after the crossfade window opened", 100,
                                                           Config::Values::SIntValueOptions{.min = 0});
    // 300 ms: covers Discord's main reflow (max 190 ms) and later update (max 255 ms) after geometry settled, as measured.
    cfgNewBufferTimeoutMs = makeShared<Config::Values::CIntValue>("plugin:hyprmorph:new_buffer_timeout_ms",
                                                                  "After the size animation ends, ms to keep the scaled old snapshot while waiting for the new-size buffer",
                                                                  300, Config::Values::SIntValueOptions{.min = 0});
    // 300 ms: hold cap. Electron chat clients have their mesh at ~245-315 ms; a 200 ms cap + the ~62 ms join window
    // misses most of them; a client whose mesh is ready earlier releases the hold then, so only slow clients pay the extra wait. 0 = no hold.
    cfgHoldMs = makeShared<Config::Values::CIntValue>("plugin:hyprmorph:hold_ms",
                                                      "Max ms a window's size/position animation is held at its start while the mesh is prepared (0 = never hold)", 300,
                                                      Config::Values::SIntValueOptions{.min = 0});
    cfgReadiness     = makeShared<Config::Values::CBoolValue>("plugin:hyprmorph:readiness", "Sample new content and run the readiness detector (0 = off)", true);
    cfgReadinessDump = makeShared<Config::Values::CStringValue>("plugin:hyprmorph:readiness_dump",
                                                                "Directory for per-resize readiness thumbnails and a replay manifest (empty = off; debugging only)", "");
    cfgPrepWorker    = makeShared<Config::Values::CStringValue>("plugin:hyprmorph:prep_worker",
                                                                "Path of a prep-worker executable to run instead of the one built into the plugin (empty = built-in)", "");
    cfgMesh       = makeShared<Config::Values::CBoolValue>("plugin:hyprmorph:mesh", "Draw the prepared mesh when it is ready before the old/new crossfade starts (0 = plain scale only)", true);
    // 80 ms: untuned starting value for the mesh weight ramp.
    cfgMeshRampMs = makeShared<Config::Values::CIntValue>("plugin:hyprmorph:mesh_ramp_ms",
                                                          "Ms over which a mesh whose content changed under it relaxes back to plain scale", 80,
                                                          Config::Values::SIntValueOptions{.min = 0});
    cfgVerbose    = makeShared<Config::Values::CBoolValue>("plugin:hyprmorph:verbose",
                                                         "Write per-resize events to the Hyprland log (0 = warnings only; counters: hyprctl hyprmorph)", false);
    for (const auto& value : {SP<Config::Values::IValue>(cfgEnabled), SP<Config::Values::IValue>(cfgCrossfadeSpan), SP<Config::Values::IValue>(cfgCrossfadeMs),
                              SP<Config::Values::IValue>(cfgCrossfadeBlurPx), SP<Config::Values::IValue>(cfgNewBufferTimeoutMs),
                              SP<Config::Values::IValue>(cfgHoldMs), SP<Config::Values::IValue>(cfgReadiness), SP<Config::Values::IValue>(cfgReadinessDump),
                              SP<Config::Values::IValue>(cfgPrepWorker), SP<Config::Values::IValue>(cfgMesh), SP<Config::Values::IValue>(cfgMeshRampMs),
                              SP<Config::Values::IValue>(cfgVerbose)})
        if (!HyprlandAPI::addConfigValueV2(handle, value))
            throw std::runtime_error(std::string{"hyprmorph: cannot register "} + value->name());

    if (std::string error; !Hold::init(error)) {
        holdDisabled = error;
        warn("animation hold disabled: " + error);
    }
    Readiness::start({});
    // At load, not at the first resize: the worker needs 0.5-2 s (context, kernels, warm-up) before its first answer is fast.
    syncPrepWorker();

    // An idle compositor renders no frame until the resize itself, so waiting for preChecks would take the new goal as baseline.
    syncWindows();
    preChecksListener = Event::bus()->m_events.render.preChecks.listen([](PHLMONITOR monitor) { onPreChecks(monitor); });
    statsCommand = HyprlandAPI::registerHyprCtlCommand(handle, IPC::Socket1::SCommand{.name    = "hyprmorph",
                                                                                      .match   = IPC::Socket1::COMMAND_MATCH_EXACT,
                                                                                      .handler = [](const IPC::Socket1::SRequest&) { return IPC::Socket1::SResponse{statsJson()}; }});
    trace("loaded");
    return {"hyprmorph", "Window resize transitions that keep content in place while the box animates", "WavJaby", "1.0.0"};
}

APICALL EXPORT void PLUGIN_EXIT() {
    if (statsCommand)
        HyprlandAPI::unregisterHyprCtlCommand(handle, statsCommand);
    statsCommand      = nullptr;
    preChecksListener = nullptr;
    Hold::exit();
    // Transformers' vtables live in this .so: every attached one must leave its window before unload.
    if (const auto gl = g_pHyprRenderer->glBackend())
        gl->makeEGLCurrent();
    for (auto& t : tracked)
        detach(*t, "");
    tracked.clear();
    // The worker runs this library's code: join before unmap. Summaries of the generations closed above are traced here.
    Readiness::stop();
    onReadinessEvents();
    Prep::stop();
    Presenter::destroy();
    analysisPool.destroy();
    staged.clear();
    Thumbnail::destroy();
}
