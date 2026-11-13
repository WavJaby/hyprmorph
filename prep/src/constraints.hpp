#pragma once
// Point constraints: reliable same-region matches fitted into each leaf's midpoint mesh. All regions' CG systems of a
// case go to the GPU in one dispatch; the RANSAC calls keep their sequential order, so the RNG sequence and thus the
// results stay as tuned (the solves do not touch the RNG).
#include "points.hpp"
#include "cg.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace prep {

struct ConstraintResult {
    int region = 0;
    bool accepted = false;
    std::string reason;
    double strength = 0, beforeMean = 0, afterMean = 0;
};

std::vector<ConstraintResult> applyPointConstraints(const cv::Mat& oldImage, const cv::Mat& newImage, std::vector<Leaf>& leaves,
                                                    const Points& source, const Points& target, CgSolver& solver);

} // namespace prep
