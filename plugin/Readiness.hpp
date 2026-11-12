// In-process readiness (S5): one readiness::Detector per window per resize generation (D1-7), run on a worker thread so the ~2.4 ms per
// observation (S3 live median) stays off the compositor thread. Scheduling = readiness/live.cpp, which the S3 live gate validated: every
// fresh thumbnail is observed, the latest one is re-observed every reobserveMs while no newer commit is pending, and a selection whose sample
// is older than the policy's sample-gap limit is vetoed when a newer commit is already pending.
// The compositor thread only queues inputs and drains events; the worker never touches Hyprland or the logger.
#pragma once

#include "Thumbnail.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace Readiness {
    struct SConfig {
        double      observeMs   = 900; // a generation is closed (summary) this long after its resize at the latest
        double      reobserveMs = 8;   // re-observation period of the latest thumbnail while no newer commit is pending
    };

    void start(const SConfig& config);
    // Joins the worker; must run before the plugin unmaps (the thread executes this library's code).
    void stop();

    // dumpDir non-empty: this generation's thumbnails + a readiness_replay manifest go there, for offline parity (readiness_replay).
    void beginGeneration(uint64_t window, uint64_t generation, uint64_t resizeNs, std::string dumpDir);
    void endGeneration(uint64_t window);
    void noteCommit(uint64_t window, uint64_t commit);
    void submit(Thumbnail::SFrame&& frame); // geometry/new-buffer evidence is implied: only frames acked at the target size are sampled

    struct SEvent {
        // CANDIDATE: a frame opened a confirm window -- the layout a SELECTED with the same candidate id will confirm, unless a change
        // restarts the window first (a new CANDIDATE). Lets the scheduler prepare the mesh while the detector is still confirming.
        enum EKind { CANDIDATE, SELECTED, VETOED, REVOKED, SUMMARY, ERROR } kind;
        uint64_t    window = 0, generation = 0;
        uint32_t    candidate = 0; // CANDIDATE, SELECTED: the detector's confirm-window counter, > 0
        double      elapsedMs = 0;
        std::string text; // SUMMARY: one JSON object; ERROR: reason
    };
    std::vector<SEvent> drain();
}
