#pragma once
// Paired rectilinear partitions: seams, best splits, baseline and joint partitions, selection evidence, midpoint layout.
#include "gutter.hpp"
#include "points.hpp"
#include "pairing.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace prep {

// An affine needs 3 correspondences; <= 2 matches cannot define a region's motion (vanishing leaves, gutter evidence).
constexpr int VANISHED_MATCHES = 2;
// Least extent either side of a cut keeps along the cut axis (line seams and gutters).
constexpr int SEAM_MARGIN = 8;

struct Seam {
    int position = 0;            // image coordinate of the line
    double relative = 0, support = 0, quality = 0;
    std::string kind;            // step / rule / faint-rule / matched-contour / gutter
    double left[3] = {0, 0, 0}, right[3] = {0, 0, 0};
    std::optional<double> landmarkError;  // joint partition only
    std::string pair;            // matched-contour only
    std::optional<int> detected;  // baseline contour snap: detected position before snapping
    std::string snap;
    std::optional<std::array<int, 2>> gutterLines;  // gutter only: its identical lines [start, end); position = their centre
    json::Value toJson() const;
};

// Seam statistics of one image. Every per-row seam flag (step / rule / faint line at a pixel)
// depends only on the 8 pixels around it along the axis, never on the queried box, so the flags are computed once per
// image and axis and a box's support is a prefix-sum difference; only lines that pass a support threshold need the
// percentile/median statistics. Exact integer arithmetic, same candidates as a per-box float computation.
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
    std::string origin;  // joint: detected-seam / matched-contour; both: anchored-gutter / matched-gutter
    double score = 0;
    json::Value toJson() const;
};

struct PartitionResult {
    std::vector<Leaf> leaves;
    std::vector<Split> splits;
    int depthLimited = 0;  // leaves at the depth bound that still had a cut: the bound, not the evidence, ended them
};

// Split evidence of both images per box pair: line seams (cached here) and gutters (cached by the finder). A gutter is offered only where a
// node has no accepted line seam.
struct SeamCache {
    const SeamIndex& oldIndex;
    const SeamIndex& newIndex;
    GutterFinder& gutters;
    std::map<std::pair<Box, Box>, std::array<std::array<std::vector<Seam>, 2>, 2>> found;  // [axis][side]
    const std::array<std::array<std::vector<Seam>, 2>, 2>& get(const Box& a, const Box& b);
};

PartitionResult baselinePartition(Shape oldShape, Shape newShape, const std::vector<Pair>& candidates, int maxDepth, SeamCache& seams);
PartitionResult jointPartition(Shape oldShape, Shape newShape, const std::vector<Pair>& contourPairs, const std::vector<Pair>& smallPairs,
                               const Points& source, const Points& target, double alignmentWeight, int maxDepth, SeamCache& seams);

struct ContourDetail {
    std::string name;
    double error = 0, weight = 0;
};
// Contour midpoint error on the default midpoint layout; nullopt when no pair has weight.
std::optional<double> contourMidpointError(const std::vector<Leaf>& leaves, const std::vector<Pair>& pairs, std::vector<ContourDetail>& details);
std::optional<double> midpointError(const std::vector<Leaf>& leaves, const Points& source, const Points& target);
std::pair<std::string, std::string> choosePreview(std::optional<double> baseContour, std::optional<double> guidedContour,
                                                  std::optional<double> baseLandmark, std::optional<double> guidedLandmark,
                                                  const std::vector<ContourDetail>& baseDetails, const std::vector<ContourDetail>& guidedDetails);

// Collapses of vanishing leaves + relayout; returns the leaves whose midpoint box changed.
std::map<std::string, int> vanishingCollapses(const std::vector<Leaf>& leaves, const std::vector<Split>& splits, const Points& source,
                                              const Points& target);
std::vector<std::string> relayout(std::vector<Leaf>& leaves, const std::vector<Split>& splits, const std::map<std::string, int>& collapse);
void defaultMidpoint(Leaf& leaf);

} // namespace prep
