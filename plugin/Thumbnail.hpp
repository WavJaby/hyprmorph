// GPU thumbnails for readiness (FrameSource seam, M1): a compute shader box-downsamples a texture region to WIDTH px wide, read back through a
// small pool of fenced PBOs so the compositor never waits on the GPU. Same filter as the S3 sampler (experiments/native_readiness_sampler.cpp,
// 4x4 bilinear taps per output pixel), whose thumbnails the readiness thresholds were derived from.
// All calls need the compositor's EGL context current and run on the compositor thread.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace Thumbnail {
    constexpr int WIDTH = 256, MAX_HEIGHT = 256;

    struct STag {
        uint64_t window = 0, generation = 0, commit = 0, sampleNs = 0;
    };
    struct SFrame {
        STag                 tag;
        int                  width = 0, height = 0;
        std::vector<uint8_t> rgba; // top row first
    };

    // Downsamples uv rect (u0, v0)-(u0+uw, v0+vh) of `texture` (v = 0 at the top row of the content) with the given content aspect (h / w).
    // false -> not submitted (no free PBO, or GL init failed: see failure()); the caller retries on a later frame.
    bool submit(unsigned texture, float u0, float v0, float uw, float vh, double aspect, const STag& tag);
    // Hands every finished readback to `done`, oldest submission first; stops at the first unfinished one.
    void poll(const std::function<void(SFrame&&)>& done);
    void destroy();

    struct SCounters {
        unsigned long submitted = 0, busy = 0, completed = 0, glErrors = 0;
    };
    const SCounters&   counters();
    const char*        failure(); // empty while working
}
