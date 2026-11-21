#include "gutter.hpp"

#include "clip.hpp"
#include "flat.hpp"
#include "mesh_checks.hpp"
#include "pairing.hpp"
#include "partition.hpp"
#include "proposals.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

namespace prep {
namespace {

struct Gutter { int start, end, centre; };  // image coordinates, lines [start, end)

// Interior flat runs of `patch` along `axis` (cut perpendicular to it): content on both sides, cut leaving `margin` on each.
std::vector<Gutter> gutters(const cv::Mat& patch, int axis, int origin, int margin) {
    std::vector<uint8_t> flat = flatIntervals(identicalIntervals(patch, axis));
    const int lines = axis ? patch.rows : patch.cols, n = int(flat.size());
    std::vector<Gutter> out;
    for (int i = 0; i < n;) {
        if (!flat[size_t(i)]) { ++i; continue; }
        int j = i;
        while (j < n && flat[size_t(j)]) ++j;
        // Intervals [i, j) join lines i..j: pixels [i, j + 1). A run touching either end is a margin, not a gutter.
        const int centre = (i + j + 1) / 2;
        if (i > 0 && j < n && centre >= margin && lines - centre >= margin) out.push_back({origin + i, origin + j + 1, origin + centre});
        i = j;
    }
    return out;
}

using Motion = std::array<int, 2>;  // content displacement old -> new, px

// Exact motions of a half under each anchor its old and new crops agree at (empty: no agreement).
std::vector<Motion> motions(const cv::Mat& oldImage, const cv::Mat& newImage, const Box& a, const Box& b) {
    std::vector<Motion> out;
    for (auto& anchors : anchoredAgreements(crop(oldImage, a), crop(newImage, b))) {
        Motion m;
        for (int k = 0; k < 2; ++k) m[size_t(k)] = anchors[size_t(k)] == Anchor::Start ? b[size_t(k)] - a[size_t(k)] : b[size_t(k) + 2] - a[size_t(k) + 2];
        if (std::find(out.begin(), out.end(), m) == out.end()) out.push_back(m);
    }
    return out;
}

std::pair<Box, Box> halves(const Box& box, int axis, int position) {
    Box low = box, high = box;
    low[size_t(axis) + 2] = position;
    high[size_t(axis)] = position;
    return {low, high};
}

struct Modes {
    std::vector<int> of;               // per match: its mode, -1 = outlier
    std::vector<cv::Point2d> centre;   // per mode: mean displacement
};

// Motion modes of a node's raw matches: greedy clusters of displacements within MATCH_RESOLUTION of a centre match (Chebyshev), the
// match with the most neighbours first. A cluster of <= VANISHED_MATCHES matches cannot define a motion: its matches are outliers.
Modes motionModes(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst) {
    const size_t n = src.size();
    std::vector<cv::Point2f> d(n);
    for (size_t i = 0; i < n; ++i) d[i] = dst[i] - src[i];
    auto near = [&](size_t i, size_t j) { return std::max(std::fabs(d[i].x - d[j].x), std::fabs(d[i].y - d[j].y)) <= MATCH_RESOLUTION; };
    Modes out{std::vector<int>(n, -1), {}};
    for (;;) {
        size_t centre = n, count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (out.of[i] >= 0) continue;
            size_t c = 0;
            for (size_t j = 0; j < n; ++j) c += out.of[j] < 0 && near(i, j);
            if (c > count) centre = i, count = c;
        }
        if (count <= size_t(VANISHED_MATCHES)) return out;  // later clusters are no larger
        cv::Point2d sum;
        for (size_t j = 0; j < n; ++j)
            if (out.of[j] < 0 && near(centre, j)) out.of[j] = int(out.centre.size()), sum += cv::Point2d(d[j]);
        out.centre.push_back(sum / double(count));
    }
}

// Raw-match evidence for a gutter pair no half agrees across, the ORB counterpart of anchored agreement. A half "moves as one" when a
// single mode with > VANISHED_MATCHES matches in it holds the majority of the half's matches (the median's breakdown point). Same gutter in
// both images: at most VANISHED_MATCHES matches cross it (outliers), and a half that moves as one carries its gutter edge with it (start
// edge with the low half, end edge with the high half, within MATCH_RESOLUTION). Cut when both halves move as one by different modes, or
// one does and the other holds a mode it lacks. A reflowing paragraph's halves hold many small modes and no majority: never cut. Returns
// the share of the node's matches in a moving-as-one half's mode, or nothing.
std::optional<double> matchEvidence(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst, const Modes& modes, int axis,
                                    const std::array<int, 3>& oldGutter, const std::array<int, 3>& newGutter) {
    const int cutOld = oldGutter[2], cutNew = newGutter[2];
    std::map<int, int> count[2];
    int total[2] = {0, 0}, crossing = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        const bool lowOld = (axis ? src[i].y : src[i].x) < cutOld, lowNew = (axis ? dst[i].y : dst[i].x) < cutNew;
        if (lowOld != lowNew) {
            if (++crossing > VANISHED_MATCHES) return std::nullopt;
            continue;
        }
        const int h = lowOld ? 0 : 1;
        ++total[h];
        if (modes.of[i] >= 0) ++count[h][modes.of[i]];
    }
    std::set<int> held[2];
    int single[2] = {-1, -1};
    for (int h = 0; h < 2; ++h)
        for (auto [m, c] : count[h])
            if (c > VANISHED_MATCHES) {
                held[h].insert(m);
                if (2 * c > total[h]) single[h] = m;
            }
    int explained = 0;
    bool cut = false, carried = false;
    for (int h = 0; h < 2; ++h) {
        if (single[h] < 0) continue;
        const std::set<int>& other = held[1 - h];
        if (std::any_of(other.begin(), other.end(), [&](int m) { return m != single[h]; })) cut = true;
        const cv::Point2d& motion = modes.centre[size_t(single[h])];
        carried |= std::fabs(newGutter[size_t(h)] - oldGutter[size_t(h)] - (axis ? motion.y : motion.x)) <= MATCH_RESOLUTION;
        explained += count[h][single[h]];
    }
    if (!cut || !carried) return std::nullopt;
    return double(explained) / (total[0] + total[1]);
}

} // namespace

GutterFinder::GutterFinder(const cv::Mat& oldImage, const cv::Mat& newImage, const Points& source, const Points& target)
    : oldImage(oldImage), newImage(newImage), source(source), target(target) {}

const std::optional<GutterCut>& GutterFinder::find(const Box& a, const Box& b) {
    auto key = std::make_pair(a, b);
    if (auto it = found.find(key); it != found.end()) return it->second;
    std::optional<GutterCut> best;
    const bool empty = a[2] <= a[0] || a[3] <= a[1] || b[2] <= b[0] || b[3] <= b[1];
    if (!empty) {
        std::vector<cv::Point2f> src, dst;
        for (size_t i = 0; i < source.size(); ++i)
            if (contained(source[i], a) && contained(target[i], b)) src.push_back(source[i]), dst.push_back(target[i]);
        const Modes modes = motionModes(src, dst);
        const double total = double(area(a)) + area(b);
        auto offer = [&](GutterCut cut) {
            if (!best || cut.score > best->score) best = std::move(cut);
        };
        for (int axis = 0; axis < 2; ++axis) {
            const size_t lo = size_t(axis), hi = size_t(axis) + 2;
            const auto oldGutters = gutters(crop(oldImage, a), axis, a[lo], SEAM_MARGIN);
            const auto newGutters = gutters(crop(newImage, b), axis, b[lo], SEAM_MARGIN);
            for (auto& go : oldGutters)
                for (auto& gn : newGutters) {
                    GutterCut cut;
                    cut.axis = axis;
                    cut.oldPosition = go.centre;
                    cut.newPosition = gn.centre;
                    cut.oldLines = {go.start, go.end};
                    cut.newLines = {gn.start, gn.end};
                    // A half can agree exactly only if the content between it and its outer edge has the same extent in both images, i.e.
                    // the gutter starts (ends) at the same distance from the node's start (end).
                    if (go.start - a[lo] == gn.start - b[lo] || a[hi] - go.end == b[hi] - gn.end) {
                        auto [lowA, highA] = halves(a, axis, go.centre);
                        auto [lowB, highB] = halves(b, axis, gn.centre);
                        auto low = motions(oldImage, newImage, lowA, lowB), high = motions(oldImage, newImage, highA, highB);
                        if (!low.empty() || !high.empty()) {
                            // Both halves move by one common displacement: they need no cut (the node may itself be a clip container).
                            if (std::any_of(low.begin(), low.end(), [&](const Motion& m) { return std::find(high.begin(), high.end(), m) != high.end(); }))
                                continue;
                            cut.origin = "anchored-gutter";
                            cut.score = 1 + ((low.empty() ? 0. : double(area(lowA)) + area(lowB)) + (high.empty() ? 0. : double(area(highA)) + area(highB))) / total;
                            offer(cut);
                            continue;
                        }
                    }
                    if (auto score = matchEvidence(src, dst, modes, axis, {go.start, go.end, go.centre}, {gn.start, gn.end, gn.centre})) {
                        cut.origin = "matched-gutter";
                        cut.score = *score;
                        offer(cut);
                    }
                }
        }
    }
    return found.emplace(key, std::move(best)).first->second;
}

} // namespace prep
