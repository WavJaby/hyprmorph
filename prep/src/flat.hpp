#pragma once
// Separable content-preserving leaf meshes; content that does not fit shrinks isotropically, also on axes
// with no flat run: a short identical-line run at the free end absorbs instead of a single-axis stretch.
#include "points.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace prep {

// Two pixels are the same colour when every channel differs by less than this (8-bit analysis images). The plugin's analysis image is an
// exact 4x4 box average, so unchanged content reproduces bit-for-bit; the margin covers GPU rounding only.
constexpr int IDENTICAL_STEP = 4;
// A flat run (blank band: background the mesh may stretch, a gutter the partition may cut) spans at least this many identical intervals;
// shorter runs are inter-glyph gaps. Tuned: the accepted flat-absorb results depend on this exact value.
constexpr int MIN_FLAT_RUN = 3;

// Per interval between adjacent lines (axis 0: columns, 1: rows) of an 8-bit BGR patch: the two lines are identical (every pixel within
// IDENTICAL_STEP).
std::vector<uint8_t> identicalIntervals(const cv::Mat& patch, int axis);
// `identicalIntervals` with runs shorter than MIN_FLAT_RUN returned to content.
std::vector<uint8_t> flatIntervals(std::vector<uint8_t> identical);

struct FlatResult {
    int region = 0;
    std::string name;
    bool accepted = false;
    double contentScale = 1;
    std::vector<std::string> unabsorbed;  // sides whose content stays anisotropic: an axis with no identical lines is forced
    std::vector<std::string> background;  // "<side>-<axis>-<attached end>": axes with no flat run whose background run absorbs
};

// Gives every leaf still drawn by plain scaling a flat-absorbing mesh, in place.
std::vector<FlatResult> applyFlatAbsorb(const cv::Mat& oldImage, const cv::Mat& newImage, std::vector<Leaf>& leaves,
                                        const Points& source, const Points& target);

} // namespace prep
