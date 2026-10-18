// Resize transition plugin, skeleton (S1-A): every window size change shows the old snapshot, then the new buffer, plain-scaled into the
// animated box, with a short crossfade. Replaces the native path's UV stretch/crop (gap on shrink, progressive crop on grow, F12).
//
// Generation trigger: GEOMETRIC_GOAL size change, cause-agnostic (dispatcher, togglefloating, relayout, fullscreen).
// Capture timing: m_realSize's begin callback only queues sendWindowSize via doLater, and m_events.resize fires on animation ticks, which may
// run inside renderMonitor. So the goal is checked (and the old snapshot captured) at the two points that precede any texture replacement
// and run outside a render pass: the window surface's precommit, and render.preChecks of every monitor.
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

    struct STracked {
        PHLWINDOWREF               window;
        Vector2D                   baselineGoal;
        CResizeTransition*         transition = nullptr; // owned by window->m_transformers; valid while window is alive and attached
        WP<CWLSurfaceResource>     surface;
        CHyprSignalListener        precommit;
    };
    // unique_ptr: precommit lambdas hold STracked*, which must survive vector growth.
    std::vector<std::unique_ptr<STracked>> tracked;

    CHyprSignalListener preChecksListener;

    // Reaches Hyprland's logger only because the Makefile builds with -fno-gnu-unique (see there).
    void log(const std::string& msg) {
        Log::logger->log(Log::DEBUG, "[resize-transition] {}", msg);
    }

    void detach(STracked& t, const std::string& reason) {
        if (!t.transition)
            return;
        if (const auto w = t.window.lock()) {
            std::erase_if(w->m_transformers, [&](const auto& tr) { return tr.get() == t.transition; });
            g_pHyprRenderer->damageWindow(w);
        }
        t.transition = nullptr;
        if (!reason.empty())
            log("native drawing: " + reason);
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
            if (!t.transition->beginGeneration(goal))
                detach(t, t.transition->m_failure);
            return;
        }
        auto transition = makeUnique<CResizeTransition>(w);
        if (!transition->beginGeneration(goal)) {
            log("native drawing: " + transition->m_failure);
            return;
        }
        t.transition = transition.get();
        w->m_transformers.emplace_back(std::move(transition));
        g_pHyprRenderer->damageWindow(w);
    }

    void attachSurface(STracked& t, PHLWINDOW w) {
        const auto surface = w->resource();
        if (surface == t.surface.lock())
            return;
        t.surface   = surface;
        t.precommit = nullptr;
        if (surface)
            t.precommit = surface->m_events.precommit.listen([raw = &t] {
                if (const auto win = raw->window.lock(); win && win->m_isMapped)
                    checkGeneration(*raw, win, "precommit");
            });
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

    void onPreChecks(PHLMONITOR monitor) {
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
            if (!t->transition->prepareFrame(config)) {
                detach(*t, t->transition->m_failure.empty() ? "transition done" : t->transition->m_failure);
                continue;
            }
            // Hyprland stops damaging the window when the size animation ends, but the crossfade, the new-buffer timeout and the final
            // "done" check all run in prepareFrame: keep one frame queued per frame while attached.
            g_pEventLoopManager->doLater([ref = t->window] {
                if (const auto win = ref.lock())
                    g_pHyprRenderer->damageWindow(win);
            });
        }
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
    for (const auto& value : {SP<Config::Values::IValue>(cfgEnabled), SP<Config::Values::IValue>(cfgCrossfadeMs), SP<Config::Values::IValue>(cfgNewBufferTimeoutMs)})
        if (!HyprlandAPI::addConfigValueV2(handle, value))
            throw std::runtime_error(std::string{"resize-transition: cannot register "} + value->name());

    // An idle compositor renders no frame until the resize itself, so waiting for preChecks would take the new goal as baseline.
    syncWindows();
    preChecksListener = Event::bus()->m_events.render.preChecks.listen([](PHLMONITOR monitor) { onPreChecks(monitor); });
    log("loaded");
    return {"resize-transition", "Plain-scale resize transition with old snapshot and crossfade", "wavjaby", "0.1"};
}

APICALL EXPORT void PLUGIN_EXIT() {
    preChecksListener = nullptr;
    // Transformers' vtables live in this .so: every attached one must leave its window before unload.
    if (const auto gl = g_pHyprRenderer->glBackend())
        gl->makeEGLCurrent();
    for (auto& t : tracked)
        detach(*t, "");
    tracked.clear();
}
