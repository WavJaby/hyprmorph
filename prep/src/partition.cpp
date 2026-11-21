#include "partition.hpp"

#include "numeric.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <future>

namespace prep {

json::Value Seam::toJson() const {
    json::Value out = json::Value::object();
    out.set("position", position);
    out.set("relative", relative);
    out.set("support", support);
    out.set("quality", quality);
    out.set("kind", kind);
    out.set("left_color", json::Array{left[0], left[1], left[2]});
    out.set("right_color", json::Array{right[0], right[1], right[2]});
    if (landmarkError) out.set("landmark_error_px", *landmarkError);
    if (!pair.empty()) out.set("pair", pair);
    if (gutterLines) out.set("gutter_lines", json::Array{(*gutterLines)[0], (*gutterLines)[1]});
    if (detected) {
        out.set("detected_position", *detected);
        out.set("snap_to_contour", snap);
    }
    return out;
}

json::Value Split::toJson() const {
    json::Value out = json::Value::object();
    out.set("name", name);
    out.set("axis", axis ? "y" : "x");
    out.set("old_parent", json::Array{oldParent[0], oldParent[1], oldParent[2], oldParent[3]});
    out.set("new_parent", json::Array{newParent[0], newParent[1], newParent[2], newParent[3]});
    out.set("old_seam", oldSeam.toJson());
    out.set("new_seam", newSeam.toJson());
    if (!origin.empty()) out.set("origin", origin);
    out.set("score", score);
    return out;
}

namespace {

constexpr double THRESHOLDS[3] = {.60, .75, .88};
const char* KINDS[3] = {"step", "rule", "faint-rule"};

inline int med3(int a, int b, int c) { return a + b + c - std::min({a, b, c}) - std::max({a, b, c}); }

// k-th smallest of a 256-bin histogram.
int orderStatistic(const int* histogram, int rank) {
    int cumulative = 0;
    for (int v = 0; v < 256; ++v) {
        cumulative += histogram[v];
        if (cumulative > rank) return v;
    }
    return 255;
}

} // namespace

namespace {

// Per-pixel line flags of one plane (the step / rule / faint-rule seam tests at that pixel),
// accumulated down the rows.
void buildPrefix(const cv::Mat& plane, std::vector<int32_t> (&prefix)[3]) {
    int lines = plane.cols, rows = plane.rows;
    for (auto& p : prefix) p.assign(size_t(rows + 1) * size_t(lines), 0);
    for (int s = 0; s < rows; ++s) {
        const uint8_t* px = plane.ptr<uint8_t>(s);
        int32_t* out[3];
        const int32_t* in[3];
        for (int k = 0; k < 3; ++k) {
            in[k] = &prefix[k][size_t(s) * size_t(lines)];
            out[k] = &prefix[k][size_t(s + 1) * size_t(lines)];
            std::copy(in[k], in[k] + lines, out[k]);
        }
        for (int g = 4; g + 3 < lines; ++g) {
            int rangeL = 0, rangeR = 0, stepDiff = 0, cl = 0, cr = 0;
            for (int c = 0; c < 3; ++c) {
                int a = px[(g - 4) * 3 + c], b = px[(g - 3) * 3 + c], d = px[(g - 2) * 3 + c];
                int e = px[(g + 1) * 3 + c], f = px[(g + 2) * 3 + c], h = px[(g + 3) * 3 + c], center = px[g * 3 + c];
                int loL = std::min(std::min(a, b), d), hiL = std::max(std::max(a, b), d);
                int loR = std::min(std::min(e, f), h), hiR = std::max(std::max(e, f), h);
                int medL = a + b + d - loL - hiL, medR = e + f + h - loR - hiR;
                rangeL = std::max(rangeL, hiL - loL);
                rangeR = std::max(rangeR, hiR - loR);
                stepDiff = std::max(stepDiff, std::abs(medL - medR));
                cl = std::max(cl, std::abs(center - medL));
                cr = std::max(cr, std::abs(center - medR));
            }
            int contrast = std::min(cl, cr), lo = std::min(rangeL, rangeR), hi = std::max(rangeL, rangeR);
            // Integer data: the faint-rule test contrast > 1.5 is contrast >= 2.
            out[0][g] += stepDiff > 5 && lo <= 12;
            out[1][g] += contrast > 3 && hi <= 12;
            out[2][g] += contrast >= 2 && hi <= 2;
        }
    }
}

} // namespace

SeamIndex::SeamIndex(const cv::Mat& source) {
    planes[0] = source;
    cv::transpose(source, planes[1]);
    auto columns = std::async(std::launch::async, [&] { buildPrefix(planes[1], prefix[1]); });
    buildPrefix(planes[0], prefix[0]);
    columns.get();
}

std::vector<Seam> SeamIndex::seams(const Box& box, int axis) const {
    int start = box[size_t(axis)], length = box[size_t(axis) + 2] - box[size_t(axis)];
    int s0 = box[size_t(1 - axis)], s1 = box[size_t(3 - axis)], span = s1 - s0;
    if (length < 2 * SEAM_MARGIN + 8 || span < 24) return {};
    const cv::Mat& plane = planes[axis];
    size_t lines = size_t(plane.cols);
    auto pixel = [&](int line, int s, int c) { return int(plane.ptr<uint8_t>(s)[line * 3 + c]); };
    std::vector<Seam> candidates;
    for (int p = SEAM_MARGIN; p <= length - SEAM_MARGIN; ++p) {
        int g = start + p;
        double supports[3];
        int kind = -1;
        for (int k = 0; k < 3; ++k) {
            const std::vector<int32_t>& prefixK = prefix[axis][k];
            supports[k] = double(prefixK[size_t(s1) * lines + size_t(g)] - prefixK[size_t(s0) * lines + size_t(g)]) / span;
            if (supports[k] >= THRESHOLDS[k] && (kind < 0 || supports[k] > supports[kind])) kind = k;
        }
        if (kind < 0) continue;
        int direct[256] = {}, left[3][256] = {}, right[3][256] = {};
        for (int s = s0; s < s1; ++s) {
            int d = 0;
            for (int c = 0; c < 3; ++c) {
                d = std::max(d, std::abs(pixel(g, s, c) - pixel(g - 1, s, c)));
                ++left[c][med3(pixel(g - 4, s, c), pixel(g - 3, s, c), pixel(g - 2, s, c))];
                ++right[c][med3(pixel(g + 1, s, c), pixel(g + 2, s, c), pixel(g + 3, s, c))];
            }
            ++direct[d];
        }
        // np.percentile(direct, 75) (linear) and np.median of the flank medians, from exact order statistics.
        double rank = .75 * (span - 1), fraction = rank - std::floor(rank);
        double lo = orderStatistic(direct, int(std::floor(rank))), hi = orderStatistic(direct, int(std::ceil(rank)));
        double strength = std::min(fraction >= .5 ? hi - (hi - lo) * (1 - fraction) : lo + (hi - lo) * fraction, 40.);
        Seam seam;
        seam.position = g;
        seam.relative = double(p) / length;
        seam.support = supports[kind];
        seam.quality = seam.support + .001 * strength;
        seam.kind = KINDS[kind];
        int mid0 = (span - 1) / 2, mid1 = span / 2;
        for (int c = 0; c < 3; ++c) {
            seam.left[c] = (orderStatistic(left[c], mid0) + orderStatistic(left[c], mid1)) / 2.;
            seam.right[c] = (orderStatistic(right[c], mid0) + orderStatistic(right[c], mid1)) / 2.;
        }
        candidates.push_back(seam);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Seam& a, const Seam& b) { return a.quality > b.quality; });
    std::vector<Seam> selected;
    for (auto& c : candidates)
        if (std::all_of(selected.begin(), selected.end(), [&](const Seam& o) { return std::abs(c.position - o.position) >= 10; }))
            selected.push_back(c);
    if (selected.size() > 8) selected.resize(8);
    std::sort(selected.begin(), selected.end(), [](const Seam& a, const Seam& b) { return a.position < b.position; });
    return selected;
}

const std::array<std::array<std::vector<Seam>, 2>, 2>& SeamCache::get(const Box& a, const Box& b) {
    auto key = std::make_pair(a, b);
    auto it = found.find(key);
    if (it != found.end()) return it->second;
    std::array<std::array<std::vector<Seam>, 2>, 2> value;
    // The joint search probes empty boxes; they have no seams.
    bool empty = a[2] <= a[0] || a[3] <= a[1] || b[2] <= b[0] || b[3] <= b[1];
    if (!empty)
        for (int axis = 0; axis < 2; ++axis) {
            value[size_t(axis)][0] = oldIndex.seams(a, axis);
            value[size_t(axis)][1] = newIndex.seams(b, axis);
        }
    return found.emplace(key, std::move(value)).first->second;
}

void defaultMidpoint(Leaf& leaf) {
    for (int k = 0; k < 4; ++k) leaf.midpointBox[size_t(k)] = (double(leaf.oldBox[size_t(k)]) + double(leaf.newBox[size_t(k)])) / 2;
}

namespace {

double colorDistance(const Seam& a, const Seam& b) {
    std::vector<double> d;
    for (int c = 0; c < 3; ++c) d.push_back(std::fabs(a.left[c] - b.left[c]));
    for (int c = 0; c < 3; ++c) d.push_back(std::fabs(a.right[c] - b.right[c]));
    return numeric::mean(d) / 255;
}

std::pair<Box, Box> splitBox(const Box& box, int axis, int position) {
    Box low = box, high = box;
    low[size_t(axis) + 2] = position;
    high[size_t(axis)] = position;
    return {low, high};
}

bool unchangedExtent(const Box& a, const Box& b) { return a[2] - a[0] == b[2] - b[0] && a[3] - a[1] == b[3] - b[1]; }

struct Option {
    double score;
    int axis;
    Seam first, second;
    std::string origin;
};

// The node's gutter cut as a split option, when it has one.
std::optional<Option> gutterOption(SeamCache& seams, const Box& a, const Box& b) {
    const auto& cut = seams.gutters.find(a, b);
    if (!cut) return std::nullopt;
    Seam first, second;
    first.position = cut->oldPosition;
    second.position = cut->newPosition;
    first.relative = double(cut->oldPosition - a[size_t(cut->axis)]) / (a[size_t(cut->axis) + 2] - a[size_t(cut->axis)]);
    second.relative = double(cut->newPosition - b[size_t(cut->axis)]) / (b[size_t(cut->axis) + 2] - b[size_t(cut->axis)]);
    first.quality = second.quality = cut->score;
    first.kind = second.kind = "gutter";
    first.gutterLines = cut->oldLines;
    second.gutterLines = cut->newLines;
    return Option{cut->score, cut->axis, first, second, cut->origin};
}

template <typename Choose>
PartitionResult partition(Shape oldShape, Shape newShape, int maxDepth, Choose choose, bool snap, const std::vector<Pair>& candidates) {
    PartitionResult out;
    struct Item { std::string name; Box a, b; int depth; };
    std::deque<Item> frontier{{"body", {0, 0, oldShape.width, oldShape.height}, {0, 0, newShape.width, newShape.height}, 0}};
    while (!frontier.empty()) {
        Item item = frontier.front();
        frontier.pop_front();
        // Rigidly translated regions already preserve their structure: split only where dimensions change. No leaf budget: a cut needs
        // evidence (seam, or a gutter whose sides move apart), and the depth bounds the tree.
        std::optional<Option> option;
        if (!unchangedExtent(item.a, item.b)) {
            option = choose(item.a, item.b);
            if (option && item.depth >= maxDepth) {
                ++out.depthLimited;
                option.reset();
            }
        }
        if (!option) {
            Leaf leaf;
            leaf.name = item.name;
            leaf.kind = "complete-region-fallback";
            leaf.oldBox = item.a;
            leaf.newBox = item.b;
            out.leaves.push_back(leaf);
            continue;
        }
        Option o = *option;
        int axis = o.axis;
        if (snap && o.first.kind != "gutter")
            // A line detector finds an edge pixel, a container box has exclusive bounds: snap near-agreeing evidence to
            // the accepted contour so a one-pixel convention difference cannot split the object.
            for (auto& c : candidates) {
                bool snapped = false;
                for (int edge : {axis, axis + 2}) {
                    int pa = c.oldBox[size_t(edge)], pb = c.newBox[size_t(edge)];
                    if (std::abs(pa - o.first.position) <= 2 && std::abs(pb - o.second.position) <= 2 &&
                        item.a[size_t(axis)] + 8 <= pa && pa <= item.a[size_t(axis) + 2] - 8 &&
                        item.b[size_t(axis)] + 8 <= pb && pb <= item.b[size_t(axis) + 2] - 8) {
                        o.first.detected = o.first.position;
                        o.first.position = pa;
                        o.first.snap = c.name;
                        o.second.detected = o.second.position;
                        o.second.position = pb;
                        o.second.snap = c.name;
                        snapped = true;
                        break;
                    }
                }
                if (snapped) break;
            }
        auto [aa, ab] = splitBox(item.a, axis, o.first.position);
        auto [ba, bb] = splitBox(item.b, axis, o.second.position);
        Split split;
        split.name = item.name;
        split.axis = axis;
        split.oldParent = item.a;
        split.newParent = item.b;
        split.oldSeam = o.first;
        split.newSeam = o.second;
        split.origin = o.origin;
        split.score = pyround(o.score, 4);
        out.splits.push_back(split);
        frontier.push_back({item.name + "-0", aa, ba, item.depth + 1});
        frontier.push_back({item.name + "-1", ab, bb, item.depth + 1});
    }
    for (auto& leaf : out.leaves) defaultMidpoint(leaf);
    return out;
}

bool within(const Box& inner, const Box& outer) {
    return outer[0] <= inner[0] && outer[1] <= inner[1] && inner[2] <= outer[2] && inner[3] <= outer[3];
}

int sideOf(const Box& box, int axis, int position) {
    if (box[size_t(axis) + 2] <= position + 1) return 0;
    if (box[size_t(axis)] >= position - 1) return 1;
    return -1;
}

double violations(int axis, int first, int second, const Box& pa, const Box& pb, const std::vector<const Pair*>& pairs) {
    double count = 0.;
    for (const Pair* pair : pairs) {
        if (!(within(pair->oldBox, pa) && within(pair->newBox, pb))) continue;
        int os = sideOf(pair->oldBox, axis, first), ns = sideOf(pair->newBox, axis, second);
        if (os < 0 || ns < 0 || os != ns) count += .06 * std::min(1., pair->evidence.score / 20);
    }
    return count;
}

// Mean residual (capped at 30 px) of trusted landmarks between the two sides' default midpoints after the split;
// float32, the precision the partition choice was tuned on.
std::optional<double> midpointAlignment(const Box& parentA, const Box& parentB, int axis, int first, int second, const Points& source,
                                        const Points& target) {
    if (source.size() < 8) return std::nullopt;
    std::vector<cv::Point2f> src, dst;
    for (size_t i = 0; i < source.size(); ++i)
        if (contained(source[i], parentA) && contained(target[i], parentB)) {
            src.push_back(source[i]);
            dst.push_back(target[i]);
        }
    if (src.size() < 8) return std::nullopt;
    std::vector<float> residual(src.size(), 30.f);
    for (int high = 0; high < 2; ++high) {
        double boxA[4], boxB[4];
        for (int k = 0; k < 4; ++k) {
            boxA[k] = parentA[size_t(k)];
            boxB[k] = parentB[size_t(k)];
        }
        if (high) {
            boxA[axis] = first;
            boxB[axis] = second;
        } else {
            boxA[axis + 2] = first;
            boxB[axis + 2] = second;
        }
        double middle[4];
        for (int k = 0; k < 4; ++k) middle[k] = (boxA[k] + boxB[k]) / 2;
        for (size_t i = 0; i < src.size(); ++i) {
            bool sa = (axis ? src[i].y : src[i].x) >= first, sb = (axis ? dst[i].y : dst[i].x) >= second;
            if (sa != sb || sa != bool(high)) continue;
            double s[2] = {src[i].x, src[i].y}, t[2] = {dst[i].x, dst[i].y}, d[2];
            for (int k = 0; k < 2; ++k) {
                double x = middle[k] + (s[k] - boxA[k]) / (boxA[k + 2] - boxA[k]) * (middle[k + 2] - middle[k]);
                double y = middle[k] + (t[k] - boxB[k]) / (boxB[k + 2] - boxB[k]) * (middle[k + 2] - middle[k]);
                d[k] = x - y;
            }
            residual[i] = float(std::min(30., std::sqrt(d[0] * d[0] + d[1] * d[1])));
        }
    }
    return double(numeric::sum(residual) / float(residual.size()));
}

} // namespace

PartitionResult baselinePartition(Shape oldShape, Shape newShape, const std::vector<Pair>& candidates, int maxDepth, SeamCache& seams) {
    auto choose = [&](const Box& a, const Box& b) -> std::optional<Option> {
        auto& evidence = seams.get(a, b);
        std::optional<Option> best;
        for (int axis = 0; axis < 2; ++axis)
            for (auto& first : evidence[size_t(axis)][0])
                for (auto& second : evidence[size_t(axis)][1]) {
                    double delta = std::fabs(first.relative - second.relative);
                    if (delta > .4) continue;
                    double color = colorDistance(first, second);
                    double kind = first.kind != second.kind ? .15 : 0.;
                    double structural = std::min(first.quality, second.quality) - .6 * delta - kind;
                    // Colours only break ties among structurally accepted seams.
                    if (structural >= .62) {
                        double score = structural - .04 * color;
                        if (!best || score > best->score) best = Option{score, axis, first, second, ""};
                    }
                }
        return best ? best : gutterOption(seams, a, b);
    };
    return partition(oldShape, newShape, maxDepth, choose, true, candidates);
}

PartitionResult jointPartition(Shape oldShape, Shape newShape, const std::vector<Pair>& contourPairs, const std::vector<Pair>& smallPairs,
                               const Points& source, const Points& target, double alignmentWeight, int maxDepth, SeamCache& seams) {
    std::vector<const Pair*> all;
    for (auto& p : contourPairs) all.push_back(&p);
    for (auto& p : smallPairs) all.push_back(&p);
    auto choose = [&](const Box& a, const Box& b) -> std::optional<Option> {
        auto& evidence = seams.get(a, b);
        std::optional<Option> best;
        auto offer = [&](Option o) {
            if (!best || o.score > best->score) best = std::move(o);
        };
        for (int axis = 0; axis < 2; ++axis) {
            for (auto& f : evidence[size_t(axis)][0])
                for (auto& s : evidence[size_t(axis)][1]) {
                    double delta = std::fabs(f.relative - s.relative);
                    if (delta > .4) continue;
                    double color = colorDistance(f, s);
                    double kind = f.kind != s.kind ? .15 : 0.;
                    double structural = std::min(f.quality, s.quality) - .6 * delta - kind;
                    if (structural < .62) continue;
                    double penalty = violations(axis, f.position, s.position, a, b, all);
                    auto alignment = midpointAlignment(a, b, axis, f.position, s.position, source, target);
                    double score = alignment ? structural - .04 * color - penalty - alignmentWeight * *alignment / 30
                                             : structural - .04 * color - penalty;
                    Seam first = f, second = s;
                    first.landmarkError = second.landmarkError = alignment;
                    offer({score, axis, first, second, "detected-seam"});
                }
            // A matched contour is itself paired boundary evidence; the seam detector can miss it when occluded.
            for (auto& pair : contourPairs) {
                if (!(within(pair.oldBox, a) && within(pair.newBox, b)) || pair.source != "direct-match") continue;
                for (int edge : {axis, axis + 2}) {
                    int pa = pair.oldBox[size_t(edge)], pb = pair.newBox[size_t(edge)];
                    if (!(a[size_t(axis)] + 8 <= pa && pa <= a[size_t(axis) + 2] - 8 && b[size_t(axis)] + 8 <= pb && pb <= b[size_t(axis) + 2] - 8))
                        continue;
                    double relA = double(pa - a[size_t(axis)]) / (a[size_t(axis) + 2] - a[size_t(axis)]);
                    double relB = double(pb - b[size_t(axis)]) / (b[size_t(axis) + 2] - b[size_t(axis)]);
                    double delta = std::fabs(relA - relB);
                    if (delta > .4) continue;
                    double structural = .95 - .6 * delta;
                    double penalty = violations(axis, pa, pb, a, b, all);
                    Seam first, second;
                    first.position = pa;
                    first.relative = relA;
                    second.position = pb;
                    second.relative = relB;
                    first.quality = second.quality = .95;
                    first.kind = second.kind = "matched-contour";
                    first.pair = second.pair = pair.name;
                    auto alignment = midpointAlignment(a, b, axis, pa, pb, source, target);
                    double score = alignment ? structural - penalty - alignmentWeight * *alignment / 30 : structural - penalty;
                    first.landmarkError = second.landmarkError = alignment;
                    offer({score, axis, first, second, "matched-contour"});
                }
            }
        }
        return best ? best : gutterOption(seams, a, b);
    };
    return partition(oldShape, newShape, maxDepth, choose, false, {});
}

namespace {

std::vector<double> linspace(double start, double stop, int num) {
    std::vector<double> out(static_cast<size_t>(num));
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) out[size_t(i)] = i * step + start;
    out.back() = stop;
    return out;
}

std::array<double, 4> mappedContourBox(const std::vector<Leaf>& leaves, const Box& box, int side) {
    auto x = linspace(box[0] + .5, box[2] - .5, 17), y = linspace(box[1] + .5, box[3] - .5, 17);
    std::vector<std::array<double, 2>> samples;
    for (double v : x) samples.push_back({v, box[1] + .5});
    for (double v : x) samples.push_back({v, box[3] - .5});
    for (double v : y) samples.push_back({box[0] + .5, v});
    for (double v : y) samples.push_back({box[2] - .5, v});
    std::vector<std::array<double, 2>> mapped(samples.size());
    for (auto& leaf : leaves) {
        const Box& s = leaf.box(side);
        const BoxF& m = leaf.midpointBox;
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto& p = samples[i];
            if (!(p[0] >= s[0] && p[0] < s[2] && p[1] >= s[1] && p[1] < s[3])) continue;
            for (int k = 0; k < 2; ++k)
                mapped[i][size_t(k)] = m[size_t(k)] + (p[size_t(k)] - s[size_t(k)]) / (s[size_t(k) + 2] - s[size_t(k)]) * (m[size_t(k) + 2] - m[size_t(k)]);
        }
    }
    std::array<double, 4> out{INFINITY, INFINITY, -INFINITY, -INFINITY};
    for (auto& p : mapped) {
        out[0] = std::min(out[0], p[0]); out[1] = std::min(out[1], p[1]);
        out[2] = std::max(out[2], p[0]); out[3] = std::max(out[3], p[1]);
    }
    return out;
}

} // namespace

std::optional<double> contourMidpointError(const std::vector<Leaf>& leaves, const std::vector<Pair>& pairs, std::vector<ContourDetail>& details) {
    double totalWeight = 0., weighted = 0.;
    details.clear();
    for (auto& pair : pairs) {
        double target[4], scale[4];
        for (int k = 0; k < 4; ++k) target[k] = (double(pair.oldBox[size_t(k)]) + pair.newBox[size_t(k)]) / 2;
        for (int k = 0; k < 2; ++k) scale[k] = scale[k + 2] = std::max(target[k + 2] - target[k], 1.);
        auto om = mappedContourBox(leaves, pair.oldBox, 0), nm = mappedContourBox(leaves, pair.newBox, 1);
        std::vector<double> terms;
        for (int k = 0; k < 4; ++k) terms.push_back((std::fabs(om[size_t(k)] - target[k]) + std::fabs(nm[size_t(k)] - target[k])) / scale[k]);
        double error = numeric::mean(terms) / 2;
        double weight = std::min(double(pair.oldBox[2] - pair.oldBox[0]) * (pair.oldBox[3] - pair.oldBox[1]),
                                 double(pair.newBox[2] - pair.newBox[0]) * (pair.newBox[3] - pair.newBox[1]));
        weighted += weight * error;
        totalWeight += weight;
        details.push_back({pair.name, pyround(error, 4), pyround(weight, 1)});
    }
    if (!totalWeight) return std::nullopt;
    return pyround(weighted / totalWeight, 4);
}

std::optional<double> midpointError(const std::vector<Leaf>& leaves, const Points& source, const Points& target) {
    if (source.empty()) return std::nullopt;
    std::vector<float> residual(source.size(), 30.f);
    for (auto& leaf : leaves) {
        float a[4], b[4], m[4];
        for (int k = 0; k < 4; ++k) {
            a[k] = float(leaf.oldBox[size_t(k)]);
            b[k] = float(leaf.newBox[size_t(k)]);
            m[k] = (a[k] + b[k]) / 2.f;
        }
        for (size_t i = 0; i < source.size(); ++i) {
            if (!(contained(source[i], leaf.oldBox) && contained(target[i], leaf.newBox))) continue;
            float s[2] = {source[i].x, source[i].y}, t[2] = {target[i].x, target[i].y}, d[2];
            for (int k = 0; k < 2; ++k) {
                float x = m[k] + (s[k] - a[k]) / (a[k + 2] - a[k]) * (m[k + 2] - m[k]);
                float y = m[k] + (t[k] - b[k]) / (b[k + 2] - b[k]) * (m[k + 2] - m[k]);
                d[k] = x - y;
            }
            residual[i] = std::min(30.f, std::sqrt(d[0] * d[0] + d[1] * d[1]));
        }
    }
    return pyround(double(numeric::sum(residual) / float(residual.size())), 2);
}

std::pair<std::string, std::string> choosePreview(std::optional<double> baseContour, std::optional<double> guidedContour,
                                                  std::optional<double> baseLandmark, std::optional<double> guidedLandmark,
                                                  const std::vector<ContourDetail>& baseDetails, const std::vector<ContourDetail>& guidedDetails) {
    if (!baseContour || !baseLandmark) return {"baseline", "insufficient_verified_structure"};
    double total = 0;
    for (auto& d : baseDetails) total += d.weight;
    double dominant = .15 * total;
    for (auto& item : baseDetails) {
        auto other = std::find_if(guidedDetails.begin(), guidedDetails.end(), [&](const ContourDetail& d) { return d.name == item.name; });
        if (item.weight >= dominant && other->error > item.error + .03) return {"baseline", "dominant_contour_regressed"};
    }
    if (*guidedContour <= .8 * *baseContour && *guidedLandmark <= .95 * *baseLandmark) return {"joint", "contour_and_landmark_improved"};
    return {"baseline", "insufficient_joint_improvement"};
}

// ---- midpoint layout ----

namespace {

// "Narrows sharply": keeps at most a quarter of its extent (e.g. a Facebook sidebar 216 -> 12).
constexpr double SHARP_RATIO = .25;

struct Tree {
    std::map<std::string, std::tuple<int, std::string, std::string>> children;
    std::map<std::string, Leaf*> byName;
    std::vector<std::string> order;  // leaf names in leaf order (insertion order, not name order; the relayout walks it)
};

Tree tree(std::vector<Leaf>& leaves, const std::vector<Split>& splits) {
    Tree t;
    for (auto& s : splits) t.children[s.name] = {s.axis, s.name + "-0", s.name + "-1"};
    for (auto& l : leaves) {
        t.byName[l.name] = &l;
        t.order.push_back(l.name);
    }
    return t;
}

BoxF nodeBox(const Tree& t, const std::string& name, int side) {
    if (auto it = t.byName.find(name); it != t.byName.end()) {
        const Box& b = it->second->box(side);
        return {double(b[0]), double(b[1]), double(b[2]), double(b[3])};
    }
    auto& [axis, first, second] = t.children.at(name);
    BoxF a = nodeBox(t, first, side), b = nodeBox(t, second, side);
    return {std::min(a[0], b[0]), std::min(a[1], b[1]), std::max(a[2], b[2]), std::max(a[3], b[3])};
}

BoxF averageBox(const Tree& t, const std::string& name) {
    BoxF o = nodeBox(t, name, 0), n = nodeBox(t, name, 1), out;
    for (int k = 0; k < 4; ++k) out[size_t(k)] = (o[size_t(k)] + n[size_t(k)]) / 2;
    return out;
}

double defaultExtent(const Tree& t, const std::string& name, int axis) {
    BoxF o = nodeBox(t, name, 0), n = nodeBox(t, name, 1);
    return ((o[size_t(axis) + 2] - o[size_t(axis)]) + (n[size_t(axis) + 2] - n[size_t(axis)])) / 2;
}

} // namespace

std::map<std::string, int> vanishingCollapses(const std::vector<Leaf>& leaves, const std::vector<Split>& splits, const Points& source,
                                              const Points& target) {
    std::map<std::string, int> parentAxis, collapse;
    for (auto& s : splits) parentAxis[s.name + "-0"] = parentAxis[s.name + "-1"] = s.axis;
    for (auto& leaf : leaves) {
        auto it = parentAxis.find(leaf.name);
        if (it == parentAxis.end()) continue;
        int axis = it->second;
        double eo = leaf.oldBox[size_t(axis) + 2] - leaf.oldBox[size_t(axis)], en = leaf.newBox[size_t(axis) + 2] - leaf.newBox[size_t(axis)];
        if (std::min(eo, en) > SHARP_RATIO * std::max(eo, en)) continue;
        int inside = 0;
        for (size_t i = 0; i < source.size(); ++i) inside += contained(source[i], leaf.oldBox) && contained(target[i], leaf.newBox);
        if (inside <= VANISHED_MATCHES) collapse[leaf.name] = axis;
    }
    return collapse;
}

std::vector<std::string> relayout(std::vector<Leaf>& leaves, const std::vector<Split>& splits, const std::map<std::string, int>& collapse) {
    Tree t = tree(leaves, splits);
    std::vector<std::string> roots;
    for (auto& name : t.order)
        if (name.find('-') == std::string::npos) roots.push_back(name);
    for (auto& [name, _] : t.children)
        if (name.find('-') == std::string::npos) roots.push_back(name);
    std::function<std::optional<double>(const std::string&, int)> desired = [&](const std::string& name, int axis) -> std::optional<double> {
        if (auto it = t.byName.find(name); it != t.byName.end()) {
            auto c = collapse.find(name);
            if (c == collapse.end() || c->second != axis) return std::nullopt;
            const Leaf& leaf = *it->second;
            return double(std::min(leaf.oldBox[size_t(axis) + 2] - leaf.oldBox[size_t(axis)], leaf.newBox[size_t(axis) + 2] - leaf.newBox[size_t(axis)]));
        }
        auto& [splitAxis, first, second] = t.children.at(name);
        if (splitAxis != axis) return std::nullopt;
        auto p0 = desired(first, axis), p1 = desired(second, axis);
        if (!p0 && !p1) return std::nullopt;
        // A collapsed descendant shrinks its whole subtree by what it gives up: the freed extent goes up the tree.
        return (p0 ? *p0 : defaultExtent(t, first, axis)) + (p1 ? *p1 : defaultExtent(t, second, axis));
    };
    std::vector<std::string> changed;
    std::function<void(const std::string&, const BoxF&)> assign = [&](const std::string& name, const BoxF& box) {
        if (auto it = t.byName.find(name); it != t.byName.end()) {
            Leaf& leaf = *it->second;
            BoxF before = leaf.midpointBox;
            leaf.midpointBox = box;
            bool close = true;
            for (int k = 0; k < 4; ++k) close &= numeric::isclose(before[size_t(k)], box[size_t(k)]);
            if (!close) changed.push_back(name);
            return;
        }
        auto& [axis, first, second] = t.children.at(name);
        auto w0 = desired(first, axis), w1 = desired(second, axis);
        if (!w0 && !w1 && box == averageBox(t, name)) {
            // Untouched subtree under an untouched parent: exact averages, not a float re-derivation of them.
            assign(first, averageBox(t, first));
            assign(second, averageBox(t, second));
            return;
        }
        double length = box[size_t(axis) + 2] - box[size_t(axis)];
        std::optional<double> wanted[2] = {w0, w1};
        double defaults[2] = {defaultExtent(t, first, axis), defaultExtent(t, second, axis)};
        double fixed = 0., free = 0.;
        for (int k = 0; k < 2; ++k) {
            if (wanted[k]) fixed += *wanted[k];
            else free += defaults[k];
        }
        // Requested extents are exact; the rest of the parent's length is shared by default proportion.
        double sizes[2];
        for (int k = 0; k < 2; ++k) sizes[k] = wanted[k] ? *wanted[k] : free ? defaults[k] * (length - fixed) / free : 0.;
        if (w0 && w1)
            for (double& s : sizes) s = s * length / fixed;
        double cut = box[size_t(axis)] + sizes[0];
        BoxF low = box, high = box;
        low[size_t(axis) + 2] = cut;
        high[size_t(axis)] = cut;
        assign(first, low);
        assign(second, high);
    };
    for (auto& root : roots) assign(root, averageBox(t, root));
    return changed;
}

} // namespace prep
