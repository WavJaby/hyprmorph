#pragma once
// Rigid-patch NCC on GLES compute (port of the torch XPU scoring in gpu_rigid_patch_probe.matches).
//
// Scores every old-strip patch (widths 6/8/12/16, step 3) against every allowed new-strip position. Instead of the
// reference's unfold + matmul (P x positions x patch-length MACs), one integer column-correlation table per allowed row
// C[r][xa][xb] = sum over patch rows and channels of a*b is built once (a matmul per row, shared by all widths) and a
// patch/position dot is a sum of `width` diagonal entries. Pixels are 8-bit, so every sum is an exact integer; the NCC
// numerator/denominators are formed in emulated 64-bit integers and only the final ratio is float -- at least as exact
// as the reference's float32 normalize + matmul.
#include "gles.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace prep {

struct PatchScore {
    int x = 0, width = 0;      // old-strip patch left edge and width
    float peak = -1, second = -1;  // best allowed score, best allowed score outside +-max(4, width/2) of bestx
    int bestx = 0, besty = 0;  // new-strip position of the peak
    int reverse = 0;           // index (within this width's valid patches) of the best patch at the peak position
};

struct StripScores {
    int height = 0, oldY = 0;
    // Valid patches (std > 12) grouped by width in (6, 8, 12, 16) order, then by x.
    std::vector<PatchScore> patches;
};

class StripNcc {
public:
    explicit StripNcc(gles::Context& context);
    ~StripNcc();
    // a, b: CV_8UC3 continuous strips (old, new), long axis = x. One dispatch set for the whole batch.
    std::vector<StripScores> score(const std::vector<std::pair<cv::Mat, cv::Mat>>& jobs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace prep
