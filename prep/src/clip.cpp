#include "clip.hpp"

#include "flat.hpp"
#include "mesh_checks.hpp"

#include <algorithm>
#include <functional>
#include <map>

namespace prep {
namespace {

bool startsWithNode(const std::string& name, const std::string& node) {
    return name == node || (name.size() > node.size() && name.compare(0, node.size(), node) == 0 && name[node.size()] == '-');
}

} // namespace

std::vector<std::array<Anchor, 2>> anchoredAgreements(const cv::Mat& a, const cv::Mat& b) {
    const int dw = b.cols - a.cols, dh = b.rows - a.rows;
    if ((dw < 0 && dh > 0) || (dw > 0 && dh < 0) || std::min({a.cols, a.rows, b.cols, b.rows}) < 1) return {};
    const int w = std::min(a.cols, b.cols), h = std::min(a.rows, b.rows);
    const auto window = [&](const cv::Mat& m, Anchor x, Anchor y) {
        return m(cv::Rect(x == Anchor::End ? m.cols - w : 0, y == Anchor::End ? m.rows - h : 0, w, h));
    };
    // An unchanged axis has one window: its end anchor is the same comparison as its start, so it agrees at both or neither.
    std::vector<std::array<Anchor, 2>> out;
    cv::Mat difference;
    for (Anchor y : {Anchor::Start, Anchor::End})
        for (Anchor x : {Anchor::Start, Anchor::End}) {
            if ((x == Anchor::End && !dw) || (y == Anchor::End && !dh)) {
                const Anchor sx = dw ? x : Anchor::Start, sy = dh ? y : Anchor::Start;
                if (std::find(out.begin(), out.end(), std::array<Anchor, 2>{sx, sy}) != out.end()) out.push_back({x, y});
                continue;
            }
            cv::absdiff(window(a, x, y), window(b, x, y), difference);
            // Per pixel: the largest channel step, then the pixels at or above IDENTICAL_STEP.
            cv::Mat largest = difference.reshape(1, int(difference.total())), changed;
            cv::reduce(largest, largest, 1, cv::REDUCE_MAX);
            cv::compare(largest, IDENTICAL_STEP, changed, cv::CMP_GE);
            const int count = cv::countNonZero(changed);
            if (count <= AGREEMENT_SLACK_PX && count <= AGREEMENT_SLACK_SHARE * double(w) * h) out.push_back({x, y});
        }
    return out;
}

std::vector<std::string> mergeClipContainers(const cv::Mat& oldImage, const cv::Mat& newImage, PartitionResult& partition) {
    std::map<std::string, const Split*> splits;
    std::map<std::string, const Leaf*> leaves;
    for (auto& s : partition.splits) splits[s.name] = &s;
    for (auto& l : partition.leaves) leaves[l.name] = &l;
    std::vector<std::string> roots;
    for (auto& [name, _] : splits)
        if (name.find('-') == std::string::npos) roots.push_back(name);
    for (auto& [name, _] : leaves)
        if (name.find('-') == std::string::npos) roots.push_back(name);

    std::vector<Leaf> found;
    std::function<void(const std::string&)> visit = [&](const std::string& name) {
        auto split = splits.find(name);
        const Box& o = split != splits.end() ? split->second->oldParent : leaves.at(name)->oldBox;
        const Box& n = split != splits.end() ? split->second->newParent : leaves.at(name)->newBox;
        // Rule 1: some axis changes length (an unchanged-size identical node is plain translation, not a container).
        const bool resized = o[2] - o[0] != n[2] - n[0] || o[3] - o[1] != n[3] - n[1];
        if (auto anchors = resized ? anchoredAgreements(crop(oldImage, o), crop(newImage, n)) : std::vector<std::array<Anchor, 2>>{};
            !anchors.empty()) {
            Leaf leaf;
            leaf.name = name;
            leaf.kind = "clip-container";
            leaf.oldBox = o;
            leaf.newBox = n;
            leaf.clip = anchors.front();
            defaultMidpoint(leaf);
            found.push_back(std::move(leaf));
            return;
        }
        if (split == splits.end()) return;
        visit(name + "-0");
        visit(name + "-1");
    };
    for (auto& root : roots) visit(root);

    std::vector<std::string> names;
    for (auto& clip : found) {
        names.push_back(clip.name);
        std::erase_if(partition.splits, [&](const Split& s) { return startsWithNode(s.name, clip.name); });
        // The clip leaf takes the place of its subtree's first leaf: leaf order is the relayout's walk order.
        auto first = std::find_if(partition.leaves.begin(), partition.leaves.end(), [&](const Leaf& l) { return startsWithNode(l.name, clip.name); });
        size_t at = size_t(first - partition.leaves.begin());
        std::erase_if(partition.leaves, [&](const Leaf& l) { return startsWithNode(l.name, clip.name); });
        partition.leaves.insert(partition.leaves.begin() + long(std::min(at, partition.leaves.size())), std::move(clip));
    }
    return names;
}

} // namespace prep
