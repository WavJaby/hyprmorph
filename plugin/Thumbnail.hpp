// Readiness thumbnails: a texture region downsampled to WIDTH px wide through Readback, tagged with the generation
// and commit it shows. The readiness thresholds were derived from thumbnails made with this filter and size; changing either shifts decisions.
// All calls need the compositor's EGL context current and run on the compositor thread.
#pragma once

#include "Readback.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace Thumbnail {
    constexpr int WIDTH = 256, MAX_HEIGHT = 256;

    struct STag {
        uint64_t window = 0, generation = 0, commit = 0, sampleNs = 0;
        // Old-content placeholder alias: the old content's region shared with the new content, at the new frame's thumbnail scale. aliasInset = mask inset
        // in thumbnail px (10 logical px, as the recorded traces' aliases had).
        bool     alias      = false;
        int      aliasInset = 0;
    };
    struct SFrame {
        STag                 tag;
        int                  width = 0, height = 0;
        std::vector<uint8_t> rgba; // top row first
    };

    // Thumbnail height for a content aspect (h / w) at WIDTH; clamped to MAX_HEIGHT as when the thresholds were derived, so tall windows are squashed.
    int  heightFor(double aspect);
    // Downsamples uv rect (u0, v0)-(u0+uw, v0+vh) of `texture` (v = 0 at the top row of the content) to width x height (<= WIDTH x MAX_HEIGHT).
    // false -> not submitted (no free PBO, or GL init failed: see failure()); the caller retries on a later frame.
    bool submit(unsigned texture, float u0, float v0, float uw, float vh, int width, int height, const STag& tag);
    // Hands every finished readback to `done`, oldest submission first; stops at the first unfinished one.
    void poll(const std::function<void(SFrame&&)>& done);
    void destroy();

    const Readback::SCounters& counters();
    const char*                failure(); // empty while working
}
