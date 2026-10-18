#pragma once
// Region pairing (ports of feature_guided_region_pairing.pair_regions / evidence, structure_midpoint.geometry_cost /
// select_safe_candidates, feature_guided_small_regions.choose_supported, joint_alignment_probe.verified_landmarks,
// render_feature_guided_refinement.overlaps_coarse / contained_pair).
#include "proposals.hpp"
#include "types.hpp"

#include <optional>
#include <string>
#include <vector>

namespace prep {

struct Evidence {
    int matches = 0, inliers = 0;
    double inlierRatio = 0, coverage = 0, boundaryIou = 0, score = 0;  // rounded to 3 decimals like the reference
    json::Value toJson() const;
};

struct Pair {
    std::string name;
    int oldIndex = 0, newIndex = 0;
    Box oldBox{}, newBox{};
    double cost = 0;                  // rounded to 4 decimals (the reference stores and reuses the rounded value)
    double geometryCost = 0, margin = 0;
    std::string source;               // "direct-match" / "geometry" (coarse); empty for small pairs
    Evidence evidence;
    std::optional<std::array<float, 4>> midpoint;  // set by select_safe_candidates (float32 like the reference)
    const Box& box(int side) const { return side ? newBox : oldBox; }
    json::Value toJson() const;
};

struct Shape {
    int width = 0, height = 0;
};

Evidence evidence(const Box& oldBox, const Box& newBox, const Points& source, const Points& target);
double geometryCost(const Box& a, const Box& b, Shape oldShape, Shape newShape);
// Coarse pairs: assignment over geometry + match evidence, ambiguity rule.
std::vector<Pair> pairRegions(const std::vector<Proposal>& oldRegions, const std::vector<Proposal>& newRegions, Shape oldShape,
                              Shape newShape, const Points& source, const Points& target);
// Reject pairs whose paths cross a larger accepted pair; sets midpoint on every returned pair. `size` = midpoint domain
// (default: the rounded average of the shapes).
std::vector<Pair> selectSafeCandidates(const std::vector<Pair>& candidates, Shape oldShape, Shape newShape,
                                       std::optional<Shape> size = std::nullopt);
std::vector<Pair> chooseSupported(const std::vector<Proposal>& oldSmall, const std::vector<Proposal>& newSmall,
                                  const Points& source, const Points& target, Shape oldShape, Shape newShape);
bool overlapsCoarse(const Pair& pair, const std::vector<Pair>& coarse);
bool containedPair(const Pair& pair, const Leaf& leaf);
void verifiedLandmarks(const Points& source, const Points& target, const std::vector<Pair>& pairs, Points& trustedSource,
                       Points& trustedTarget);

// Pair with its midpoint box, the candidate_with_midpoint rule; `size` is the midpoint domain.
Pair candidateWithMidpoint(const Pair& original, Shape oldShape, Shape newShape, Shape size);
bool conflicts(const Pair& a, const Pair& b);
bool contained(const cv::Point2f& p, const Box& box);
int pyroundInt(double value);  // Python round() to int (half to even)

} // namespace prep
