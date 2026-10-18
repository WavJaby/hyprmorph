#pragma once
// Paired rectilinear partitions (ports of partition_transition.seams / best_split / paired_partition,
// joint_partition.py, joint_alignment_probe.select_partition evidence, midpoint_layout.py).
#include "pairing.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace prep {

struct Seam {
    int position = 0;            // image coordinate of the line
    double relative = 0, support = 0, quality = 0;
    std::string kind;            // step / rule / faint-rule / matched-contour
    double left[3] = {0, 0, 0}, right[3] = {0, 0, 0};
    std::optional<double> landmarkError;  // joint partition only
    std::string pair;            // matched-contour only
    std::optional<int> detected;  // baseline contour snap: detected position before snapping
    std::string snap;
    json::Value toJson() const;
};

// Seam statistics of one image. Every per-row flag of partition_transition.seams (step / rule / faint line at a pixel)
// depends only on the 8 pixels around it along the axis, never on the queried box, so the flags are computed once per
// image and axis and a box's support is a prefix-sum difference; only lines that pass a support threshold need the
// percentile/median statistics. Exact integer arithmetic, same candidates as the NumPy reference.
class SeamIndex {
public:
    explicit SeamIndex(const cv::Mat& image);
    std::vector<Seam> seams(const Box& box, int axis) const;

private:
    // [axis]: the image itself (axis 0, lines = columns) or its transpose (axis 1, lines = rows), so both axes scan
    // contiguous rows: plane row s = one perpendicular coordinate, column g = one candidate line.
    cv::Mat planes[2];
    // [axis][kind]: prefix count down the plane rows, laid out [(s) * lines + g] for s in 0..rows (row 0 = zeros).
    std::vector<int32_t> prefix[2][3];
};

struct Split {
    std::string name;
    int axis = 0;
    Box oldParent{}, newParent{};
    Seam oldSeam, newSeam;
    std::string origin;  // joint only: detected-seam / matched-contour
    double score = 0;
    json::Value toJson() const;
};

struct PartitionResult {
    std::vector<Leaf> leaves;
    std::vector<Split> splits;
};

struct SeamCache {
    const SeamIndex& oldIndex;
    const SeamIndex& newIndex;
    std::map<std::pair<Box, Box>, std::array<std::array<std::vector<Seam>, 2>, 2>> found;  // [axis][side]
    const std::array<std::array<std::vector<Seam>, 2>, 2>& get(const Box& a, const Box& b);
};

PartitionResult baselinePartition(Shape oldShape, Shape newShape, const std::vector<Pair>& candidates, int maximum, SeamCache& seams);
PartitionResult jointPartition(Shape oldShape, Shape newShape, const std::vector<Pair>& contourPairs, const std::vector<Pair>& smallPairs,
                               const Points& source, const Points& target, double alignmentWeight, int maximum, SeamCache& seams);

struct ContourDetail {
    std::string name;
    double error = 0, weight = 0;
};
// contour_midpoint_error on the default midpoint layout; nullopt when no pair has weight.
std::optional<double> contourMidpointError(const std::vector<Leaf>& leaves, const std::vector<Pair>& pairs, std::vector<ContourDetail>& details);
std::optional<double> midpointError(const std::vector<Leaf>& leaves, const Points& source, const Points& target);
std::pair<std::string, std::string> choosePreview(std::optional<double> baseContour, std::optional<double> guidedContour,
                                                  std::optional<double> baseLandmark, std::optional<double> guidedLandmark,
                                                  const std::vector<ContourDetail>& baseDetails, const std::vector<ContourDetail>& guidedDetails);

// midpoint_layout.vanishing_collapses + relayout; returns the leaves whose midpoint box changed.
std::map<std::string, int> vanishingCollapses(const std::vector<Leaf>& leaves, const std::vector<Split>& splits, const Points& source,
                                              const Points& target);
std::vector<std::string> relayout(std::vector<Leaf>& leaves, const std::vector<Split>& splits, const std::map<std::string, int>& collapse);
void defaultMidpoint(Leaf& leaf);

} // namespace prep
