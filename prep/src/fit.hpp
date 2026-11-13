#pragma once
// Local contour fits inside partition leaves.
// Weighted Jacobi solve on purpose: the accepted meshes were tuned on its early-stopped iterate; a direct solver
// differs by up to .06 px.
#include "pairing.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace prep {

struct RefineReport {
    std::string region;
    std::vector<std::string> accepted;
};

// Fits reliable interior landmarks of each leaf without coupling regions; sets leaf.meshes where any is accepted.
std::vector<RefineReport> refineRegions(const cv::Mat& oldImage, const cv::Mat& newImage, std::vector<Leaf>& leaves,
                                        const std::vector<Pair>& candidates);

} // namespace prep
