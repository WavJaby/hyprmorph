#pragma once
// Local contour fits inside partition leaves (ports of partition_transition.refine_regions, structure_midpoint.prepare /
// prepare_side / solve_reference / finish_side / source_grid, shared_mesh_midpoint.structural_lines).
// Reference Jacobi solve on purpose: its early-stopped iterate is what the d41 leaves hold (the direct solver of
// reduced-mesh-solvers.md differs by up to .06 px), and parity comes first.
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
