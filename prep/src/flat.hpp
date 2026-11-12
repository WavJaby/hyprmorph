#pragma once
// Separable content-preserving leaf meshes, D4-1 isotropic shrink (port of experiments/flat_absorb.py).
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
    std::vector<std::string> unabsorbed;
};

// Gives every leaf still drawn by plain scaling a flat-absorbing mesh, in place.
std::vector<FlatResult> applyFlatAbsorb(const cv::Mat& oldImage, const cv::Mat& newImage, std::vector<Leaf>& leaves,
                                        const Points& source, const Points& target);

} // namespace prep
