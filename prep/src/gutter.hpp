#pragma once
// Gutter splits: rows or columns separated only by blank space. A gutter is an interior flat run (identical lines, flat.hpp) across a
// partition box in both images; it becomes a cut only when the content on its two sides moves differently -- the halves' anchored
// agreement (clip.hpp) gives each half's exact motion, raw ORB matches give it where no half agrees. Text paragraphs are full of gutters
// whose halves reflow together: they never qualify, so the partition does not shatter them.
#include "points.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <map>
#include <array>
#include <optional>
#include <string>
#include <utility>

namespace prep {

struct GutterCut {
    int axis = 0;
    int oldPosition = 0, newPosition = 0;   // gutter centres: the cut in each image
    std::array<int, 2> oldLines{}, newLines{};  // the gutter's identical lines [start, end) in image coordinates
    std::string origin;  // split origin: "anchored-gutter" (exact motion of >= 1 half) / "matched-gutter" (ORB motion modes)
    // anchored-gutter: 1 + share of the node's area (old + new) whose motion is exact. matched-gutter: share of the node's matches in a
    // moving-as-one half's mode, in (0, 1]. Anchored always outranks matched.
    double score = 0;
};

class GutterFinder {
public:
    // `source` / `target`: raw ORB matches.
    GutterFinder(const cv::Mat& oldImage, const cv::Mat& newImage, const Points& source, const Points& target);
    // Best gutter cut of the node (old box a, new box b), or nothing. Cached per box pair.
    const std::optional<GutterCut>& find(const Box& a, const Box& b);

private:
    const cv::Mat& oldImage;
    const cv::Mat& newImage;
    const Points& source;
    const Points& target;
    std::map<std::pair<Box, Box>, std::optional<GutterCut>> found;
};

} // namespace prep
