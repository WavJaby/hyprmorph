#pragma once
// ORB correspondences + contour proposals (large regions, open outlines, small candidates).
#include "points.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace prep {

// ORB 1500 features, edgeThreshold 15, FAST 10, Hamming kNN ratio .8 both ways + mutual check.
void matchImages(const cv::Mat& oldImage, const cv::Mat& newImage, Points& source, Points& target);

struct Proposal {
    Box box{};
    double score = 0;   // boundary_score (rounded to 3 decimals; thresholds were tuned on the rounded values)
    std::string method;
    json::Value toJson() const;
};

struct ImageProposals {
    std::vector<Proposal> regions;  // discover(image, image)
    std::vector<Proposal> small;    // small_proposals(image, 12, 200)
};

// One pass over the image: the two Gaussian/Canny contour sets feed both the large and the small proposals.
ImageProposals propose(const cv::Mat& image);

int area(const Box& b);
int intersection(const Box& a, const Box& b);
double iou(const Box& a, const Box& b);
double pyround(double value, int digits);  // round to `digits` decimals, half to even on the double's exact value

} // namespace prep
