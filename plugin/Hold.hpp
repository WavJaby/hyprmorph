// Animation hold (D3-3): keeps a window's size/position animation at the progress it had when the generation began, so mesh preparation
// can finish before the box moves; on release the animation plays its full duration from there (total = hold + duration).
// Mechanism: libhyprutils' own GOT slot for CBaseAnimatedVariable::getPercent points at our function. hyprutils' bezier step reads progress
// only through that slot (getCurveStep: call *GOT, not inlined -- objdump of libhyprutils 0.14.3), so no Hyprland state is written. Border
// and content both read m_realSize/m_realPosition, so they stay in sync (D3-2). Spring curves step without getPercent -> never held.
// Not Hyprland's createFunctionHook: its trampoline jumps are rel32 and the library sits > 2 GB from the trampoline page ("source and
// trampo are over 2GB apart", nested run 2026-10-30).
#pragma once

#include <hyprland/src/desktop/DesktopTypes.hpp>

#include <chrono>
#include <string>

namespace Hold {
    using Clock = std::chrono::steady_clock;

    // Installs the hook. false -> reason in `error`; the plugin then runs without holds.
    bool init(std::string& error);
    // Removes the hook. A variable still held or shifted at unload loses the shift: its animation jumps forward by the hold.
    void exit();

    // Holds the window's animating size/position variables at their current position until `release` or `now + cap`, whichever first.
    // A retarget of a still-running motion inherits its deadline. false -> nothing held; reason in `why`.
    bool begin(const PHLWINDOW& window, Clock::time_point now, Clock::duration cap, std::string& why);
    // Ends the hold early (mesh ready, or the transition gave up). The animation keeps its shifted clock until it finishes.
    void release(const PHLWINDOW& window, Clock::time_point now);
    // True while any of the window's variables is still held (cap not reached, not released).
    bool held(const PHLWINDOW& window, Clock::time_point now);

    // Per window except adopted / staleDropped (per variable). retargets are counted inside holds. adopted = re-begun by a timer tick
    // while held, before begin() saw the new goal.
    struct SCounters {
        unsigned long holds = 0, retargets = 0, releasedEarly = 0, refusedSpent = 0, refusedSpring = 0, refusedIdle = 0, adopted = 0, staleDropped = 0;
    };
    const SCounters& counters();
}
