#include "Hold.hpp"

#include <hyprland/src/desktop/view/window/Window.hpp>
#include <hyprland/src/helpers/AnimatedVariable.hpp>
#include <hyprutils/animation/AnimatedVariable.hpp>

#include <dlfcn.h>
#include <elf.h>
#include <limits.h>
#include <link.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <vector>

#ifndef HOLD_HYPRUTILS_VERSION
#error "HOLD_HYPRUTILS_VERSION must name the hyprutils version whose getCurveStep -> getPercent call was checked (Makefile sets it)"
#endif

using namespace std::chrono;

namespace {
    using Base         = Hyprutils::Animation::CBaseAnimatedVariable;
    using FnGetPercent = float (*)(const Base*);

    constexpr const char* GET_PERCENT_SYMBOL = "_ZNK9Hyprutils9Animation21CBaseAnimatedVariable10getPercentEv";

    struct SEntry {
        PHLWINDOWREF            window;
        PHLANIMVARREF<Vector2D> var;
        const Base*             key = nullptr;
        // Identity of the held motion; a mismatch = the variable was re-begun (see hkGetPercent / begin for what that means).
        Vector2D                begun, goal;
        float                   percent = 0.F; // progress frozen at begin
        Hold::Clock::time_point releaseAt;     // cap, moved earlier by release(); progress restarts from `percent` here
    };

    // libhyprutils' own GOT slot for getPercent and the value it held (the real function).
    FnGetPercent*       slot     = nullptr;
    FnGetPercent        original = nullptr;
    std::vector<SEntry> entries;
    Hold::SCounters     counts;

    // Mirrors the stock getPercent: progress per ms = 1 / (internalSpeed * 100); no config -> the stock function returns 1.
    float msPerFullProgress(const Base* var) {
        const auto config = var->getConfig().lock();
        const auto values = config ? config->pValues.lock() : nullptr;
        return values ? values->internalSpeed * 100.F : 0.F;
    }

    float hkGetPercent(const Base* self) {
        if (entries.empty())
            return original(self);
        const auto it = std::ranges::find(entries, self, &SEntry::key);
        if (it == entries.end())
            return original(self);

        if (it->var.expired()) {
            ++counts.staleDropped;
            entries.erase(it);
            return original(self);
        }

        const auto now = Hold::Clock::now();
        if (it->var->goal() != it->goal || it->var->begun() != it->begun) {
            // Re-begun before begin() saw it: Hyprland's animation tick runs on a timer and can step between the new goal and the next
            // render.preChecks. Still held -> adopt the new motion at its start, same deadline (begin() then re-creates it).
            if (now < it->releaseAt) {
                ++counts.adopted;
                it->begun   = it->var->begun();
                it->goal    = it->var->goal();
                it->percent = 0.F;
                return 0.F;
            }
            // Released -> stock clock. Re-begun at the old goal = that motion had finished and this is an unrelated one (a move with
            // no resize): its budget is history, forget it -- kept, the next resize would look like a retarget of a spent motion and
            // get no hold. Re-begun elsewhere = a retarget mid-motion: follow it, so begin() still sees the spent budget.
            if (it->var->begun() == it->goal) {
                entries.erase(it);
                return original(self);
            }
            it->begun = it->var->begun();
            it->goal  = it->var->goal();
            return original(self);
        }

        if (now < it->releaseAt)
            return it->percent;

        const float span     = msPerFullProgress(self);
        const float progress = span > 0.F ? it->percent + duration<float, std::milli>(now - it->releaseAt).count() / span : 1.F;
        if (progress >= 1.F) {
            entries.erase(it);
            return 1.F;
        }
        return progress;
    }

    // Writes one GOT slot. Full RELRO (libhyprutils is linked BIND_NOW) leaves it read-only after relocation.
    bool writeSlot(FnGetPercent value) {
        const long page  = sysconf(_SC_PAGESIZE);
        auto*      start = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(slot) & ~uintptr_t(page - 1));
        if (mprotect(start, page, PROT_READ | PROT_WRITE))
            return false;
        __atomic_store_n(slot, value, __ATOMIC_SEQ_CST);
        mprotect(start, page, PROT_READ);
        return true;
    }

    struct SSearch {
        std::string   path;
        FnGetPercent* slot = nullptr;
    };

    // Finds libhyprutils' relocation for getPercent (R_X86_64_GLOB_DAT here; JUMP_SLOT accepted too) -> address of its GOT slot.
    int findSlot(dl_phdr_info* info, size_t, void* data) {
        auto* search = static_cast<SSearch*>(data);
        if (!info->dlpi_name || !std::string_view{info->dlpi_name}.contains("libhyprutils.so"))
            return 0;
        search->path    = info->dlpi_name;
        const auto base = info->dlpi_addr;
        // glibc rewrites d_ptr to absolute addresses on x86_64; keep raw vaddrs working too.
        const auto abs = [base](ElfW(Addr) p) { return p < base ? base + p : p; };

        const ElfW(Dyn)* dyn = nullptr;
        for (int i = 0; i < info->dlpi_phnum; ++i)
            if (info->dlpi_phdr[i].p_type == PT_DYNAMIC)
                dyn = reinterpret_cast<const ElfW(Dyn)*>(base + info->dlpi_phdr[i].p_vaddr);
        if (!dyn)
            return 1;

        const ElfW(Sym)*  symtab = nullptr;
        const char*       strtab = nullptr;
        const ElfW(Rela)* tables[2]{};
        size_t            sizes[2]{};
        for (auto* d = dyn; d->d_tag != DT_NULL; ++d) {
            switch (d->d_tag) {
                case DT_SYMTAB: symtab = reinterpret_cast<const ElfW(Sym)*>(abs(d->d_un.d_ptr)); break;
                case DT_STRTAB: strtab = reinterpret_cast<const char*>(abs(d->d_un.d_ptr)); break;
                case DT_RELA: tables[0] = reinterpret_cast<const ElfW(Rela)*>(abs(d->d_un.d_ptr)); break;
                case DT_RELASZ: sizes[0] = d->d_un.d_val; break;
                case DT_JMPREL: tables[1] = reinterpret_cast<const ElfW(Rela)*>(abs(d->d_un.d_ptr)); break;
                case DT_PLTRELSZ: sizes[1] = d->d_un.d_val; break;
                default: break;
            }
        }
        if (!symtab || !strtab)
            return 1;
        for (int t = 0; t < 2; ++t)
            for (size_t i = 0; tables[t] && i < sizes[t] / sizeof(ElfW(Rela)); ++i) {
                const auto& r    = tables[t][i];
                const auto  type = ELF64_R_TYPE(r.r_info);
                if ((type == R_X86_64_GLOB_DAT || type == R_X86_64_JUMP_SLOT) && std::string_view{strtab + symtab[ELF64_R_SYM(r.r_info)].st_name} == GET_PERCENT_SYMBOL)
                    search->slot = reinterpret_cast<FnGetPercent*>(base + r.r_offset);
            }
        return 1;
    }
}

bool Hold::init(std::string& error) {
    SSearch search;
    dl_iterate_phdr(findSlot, &search);
    if (search.path.empty()) {
        error = "libhyprutils not loaded";
        return false;
    }
    char real[PATH_MAX];
    const std::string expected = std::string{"libhyprutils.so."} + HOLD_HYPRUTILS_VERSION;
    if (!realpath(search.path.c_str(), real) || !std::string_view{real}.ends_with(expected)) {
        error = search.path + " is not " + expected + ", the only version whose getCurveStep -> getPercent call was checked";
        return false;
    }
    if (!search.slot) {
        error = "no GOT relocation for getPercent in " + search.path;
        return false;
    }
    // A slot not pointing into libhyprutils is already redirected (a second copy of this plugin): chaining onto it would leave a dangling
    // pointer behind whichever copy unloads first.
    Dl_info owner{};
    if (!dladdr(reinterpret_cast<void*>(*search.slot), &owner) || !owner.dli_fname || !realpath(owner.dli_fname, real) ||
        !std::string_view{real}.ends_with(expected)) {
        error = "getPercent GOT slot already redirected";
        return false;
    }
    slot     = search.slot;
    original = *slot;
    if (!writeSlot(&hkGetPercent)) {
        slot = nullptr;
        error = "cannot make the GOT page writable";
        return false;
    }
    return true;
}

void Hold::exit() {
    entries.clear();
    if (slot && *slot == &hkGetPercent)
        writeSlot(original);
    slot = nullptr;
}

bool Hold::begin(const PHLWINDOW& window, Clock::time_point now, Clock::duration cap, std::string& why) {
    if (!slot) {
        why = "hold hook not installed";
        return false;
    }
    // A resize that retargets a motion still in progress inherits its deadline, so rapid resizes freeze the box for <= cap in total, not
    // cap per resize. In progress = still held, or re-begun away from its goal (Hyprland re-begins from the current value, so a finished
    // motion re-begins exactly at its old goal). Entry presence alone is not enough: Hyprland ends an animation by warp once value == goal,
    // before progress reaches 1, so finished entries are never erased by the hook.
    auto deadline = now + cap;
    bool retarget = false;
    for (const auto& e : entries) {
        if (e.window.lock() != window || e.var.expired())
            continue;
        const bool held = now < e.releaseAt, midMotion = e.var->begun() != e.goal;
        if (held || midMotion) {
            deadline = retarget ? std::min(deadline, e.releaseAt) : e.releaseAt;
            retarget = true;
        }
    }
    std::erase_if(entries, [&](const SEntry& e) { return e.window.expired() || e.window.lock() == window; });
    if (retarget && deadline <= now) {
        ++counts.refusedSpent;
        why = "hold budget spent by the motion this resize retargets";
        return false;
    }
    if (retarget)
        ++counts.retargets;

    std::vector<PHLANIMVAR<Vector2D>*> vars;
    for (auto* var : {&window->sizeAnimation(), &window->positionAnimation()})
        if (*var && (*var)->isBeingAnimated())
            vars.push_back(var);
    if (vars.empty()) {
        ++counts.refusedIdle;
        why = "no size/position animation running";
        return false;
    }
    for (const auto* var : vars)
        if ((*var)->isSpringCurve()) {
            ++counts.refusedSpring;
            why = "spring curve steps without getPercent";
            return false;
        }

    // No tick since begin (value still == begun) -> hold at 0, exactly where the box is. Otherwise a timer tick already moved it: hold at
    // the stock progress, one small step ahead, rather than jump back.
    for (auto* var : vars)
        entries.push_back(SEntry{.window    = window,
                                 .var       = *var,
                                 .key       = var->get(),
                                 .begun     = (*var)->begun(),
                                 .goal      = (*var)->goal(),
                                 .percent   = (*var)->value() == (*var)->begun() ? 0.F : original(var->get()),
                                 .releaseAt = deadline});
    ++counts.holds;
    return true;
}

void Hold::release(const PHLWINDOW& window, Clock::time_point now) {
    bool released = false;
    for (auto& e : entries)
        if (e.window.lock() == window && e.releaseAt > now) {
            e.releaseAt = now;
            released    = true;
        }
    if (released)
        ++counts.releasedEarly;
}

bool Hold::held(const PHLWINDOW& window, Clock::time_point now) {
    return std::ranges::any_of(entries, [&](const SEntry& e) { return e.window.lock() == window && now < e.releaseAt; });
}

const Hold::SCounters& Hold::counters() {
    return counts;
}
