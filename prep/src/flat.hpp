#pragma once
// Separable content-preserving leaf meshes; content that does not fit shrinks isotropically, also on axes
// with no flat run: a short identical-line run at the free end absorbs instead of a single-axis stretch.
#include "points.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace prep {

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
