#pragma once
// Clip containers (M4 D4-4): a split-tree node whose old and new content agree pixel-for-pixel from one anchored end over their overlap
// (lists, chat logs, sidebars: shift 0, scale 1) becomes one leaf drawn 1:1 from the larger image, cropped to its box. Rows slide in on
// grow and out under the edge on shrink; nothing in it stretches or crossfades.
#include "partition.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace prep {

// Every anchor pair (x, y) under which `a` and `b` agree on every pixel of their overlap (each channel within IDENTICAL_STEP), in the order
// (start, start), (end, start), (start, end), (end, end). Overlap = the smaller crop placed at the anchor in the larger. Empty when sizes
// change in opposite directions on the two axes (no single larger image). An unchanged axis has one window, so it agrees at both of its
// anchors or neither: an unchanged-size identical pair agrees at all four. Shared by clip containers and gutter splits.
std::vector<std::array<Anchor, 2>> anchoredAgreements(const cv::Mat& a, const cv::Mat& b);

// Top-down over the chosen partition: the first node (split parent or leaf) that is a clip container replaces its whole subtree --
// leaves and splits -- with one leaf named after the node, kind "clip-container", Leaf::clip set, midpoint = average box. Run before the
// midpoint relayout. Returns the names of the clip leaves.
std::vector<std::string> mergeClipContainers(const cv::Mat& oldImage, const cv::Mat& newImage, PartitionResult& partition);

} // namespace prep
