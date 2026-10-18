#pragma once
// Mesh safety checks shared by every stage (ports of full_window_fallback.minimum_path_area,
// structure_midpoint.deformation_groups / rectangle_error, partition_transition.audit_leaf).
#include "types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace prep {

// Analytic minimum signed triangle area along the linear path old -> new grid; coordinates are taken as float32
// (every reference call site holds float32 grids), arithmetic in double.
double minimumPathArea(const Grid& from, const Grid& to);

struct Protection {
    int rows = 0, cols = 0;              // cells
    std::vector<uint8_t> cell;          // (rows) x (cols), 1 = protected (texture: must stay rectangular)
    std::vector<int> groups[2];         // per vertex, per dimension: coordinate group label (np.unique inverse)
    bool any() const;
};

// flat_only = true everywhere in the chain. `patch` is the leaf's image crop.
Protection deformationGroups(const cv::Mat& patch, const Grid& source, bool flatOnly = true);

// Max deviation from rectangular over protected cells, in float32 like the reference; 0 when none.
double rectangleError(const Grid& grid, const Protection& protection);

// audit_leaf: positive boxes, no fold along the path, protected cells rectangular. False on any violation.
bool auditLeaf(const cv::Mat& oldImage, const cv::Mat& newImage, const Leaf& leaf);

cv::Mat crop(const cv::Mat& image, const Box& box);

} // namespace prep
