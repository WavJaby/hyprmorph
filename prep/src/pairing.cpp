#include "pairing.hpp"

#include <opencv2/geometry.hpp>

#include <algorithm>
#include <cmath>
#include <map>

namespace prep {

bool contained(const cv::Point2f& p, const Box& box) {
    return p.x >= box[0] && p.x < box[2] && p.y >= box[1] && p.y < box[3];
}

int pyroundInt(double value) { return int(std::nearbyint(value)); }  // default FE_TONEAREST = half to even

json::Value Evidence::toJson() const {
    json::Value out = json::Value::object();
    out.set("matches", matches);
    out.set("inliers", inliers);
    out.set("inlier_ratio", inlierRatio);
    out.set("coverage", coverage);
    out.set("boundary_iou", boundaryIou);
    out.set("score", score);
    return out;
}

json::Value Pair::toJson() const {
    json::Value out = json::Value::object();
    out.set("name", name);
    out.set("old_index", oldIndex);
    out.set("new_index", newIndex);
    out.set("old_box", json::Array{oldBox[0], oldBox[1], oldBox[2], oldBox[3]});
    out.set("new_box", json::Array{newBox[0], newBox[1], newBox[2], newBox[3]});
    out.set("cost", cost);
    if (!source.empty()) {
        out.set("geometry_cost", geometryCost);
        out.set("ambiguity_margin", margin);
        out.set("source", source);
    }
    out.set("evidence", evidence.toJson());
    if (midpoint) out.set("midpoint_box", json::Array{double((*midpoint)[0]), double((*midpoint)[1]), double((*midpoint)[2]), double((*midpoint)[3])});
    return out;
}

namespace {

void inside(const Box& oldBox, const Box& newBox, const Points& source, const Points& target, Points& a, Points& b,
            std::vector<int>* index = nullptr) {
    for (size_t i = 0; i < source.size(); ++i)
        if (contained(source[i], oldBox) && contained(target[i], newBox)) {
            a.push_back(source[i]);
            b.push_back(target[i]);
            if (index) index->push_back(int(i));
        }
}

double areaF(const std::array<double, 4>& b) { return std::max(0., b[2] - b[0]) * std::max(0., b[3] - b[1]); }

} // namespace

Evidence evidence(const Box& oldBox, const Box& newBox, const Points& source, const Points& target) {
    Points a, b;
    inside(oldBox, newBox, source, target, a, b);
    Evidence out;
    out.matches = int(a.size());
    if (a.size() < 6) return out;
    cv::Mat mask;
    cv::Mat affine = cv::estimateAffinePartial2D(a, b, mask, cv::RANSAC, 2.5);
    if (affine.empty()) return out;
    int count = cv::countNonZero(mask);
    double ratio = double(count) / double(a.size());
    if (count < 6 || ratio < .7) return out;
    float lo[2][2] = {{INFINITY, INFINITY}, {INFINITY, INFINITY}}, hi[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}};
    for (size_t i = 0; i < a.size(); ++i) {
        if (!mask.at<uint8_t>(int(i))) continue;
        for (int s = 0; s < 2; ++s) {
            const cv::Point2f& p = s ? b[i] : a[i];
            lo[s][0] = std::min(lo[s][0], p.x); hi[s][0] = std::max(hi[s][0], p.x);
            lo[s][1] = std::min(lo[s][1], p.y); hi[s][1] = std::max(hi[s][1], p.y);
        }
    }
    double coverage = INFINITY;
    for (int s = 0; s < 2; ++s) {
        const Box& box = s ? newBox : oldBox;
        double sx = double(hi[s][0] - lo[s][0]) / std::max(box[2] - box[0], 1);
        double sy = double(hi[s][1] - lo[s][1]) / std::max(box[3] - box[1], 1);
        coverage = std::min(coverage, sx * sy);
    }
    const double* m = affine.ptr<double>(0);
    const double* n = affine.ptr<double>(1);
    float corners[4][2] = {{float(oldBox[0]), float(oldBox[1])}, {float(oldBox[2]), float(oldBox[1])},
                           {float(oldBox[2]), float(oldBox[3])}, {float(oldBox[0]), float(oldBox[3])}};
    std::array<double, 4> predicted{INFINITY, INFINITY, -INFINITY, -INFINITY};
    for (auto& c : corners) {
        double x = (double(c[0]) * m[0] + double(c[1]) * m[1]) + m[2];
        double y = (double(c[0]) * n[0] + double(c[1]) * n[1]) + n[2];
        predicted[0] = std::min(predicted[0], x); predicted[1] = std::min(predicted[1], y);
        predicted[2] = std::max(predicted[2], x); predicted[3] = std::max(predicted[3], y);
    }
    std::array<double, 4> nb{double(newBox[0]), double(newBox[1]), double(newBox[2]), double(newBox[3])};
    double overlap = areaF({std::max(predicted[0], nb[0]), std::max(predicted[1], nb[1]), std::min(predicted[2], nb[2]), std::min(predicted[3], nb[3])});
    double boundaryIou = overlap / std::max(1., areaF(predicted) + areaF(nb) - overlap);
    // Low texture density is valid; points must still span the box and agree with its contour.
    double score = coverage >= .06 && boundaryIou >= .35 ? count * std::sqrt(coverage) * boundaryIou : 0.;
    out.inliers = count;
    out.inlierRatio = pyround(ratio, 3);
    out.coverage = pyround(coverage, 3);
    out.boundaryIou = pyround(boundaryIou, 3);
    out.score = pyround(score, 3);
    return out;
}

double geometryCost(const Box& a, const Box& b, Shape oldShape, Shape newShape) {
    double os[2] = {double(oldShape.width), double(oldShape.height)}, ns[2] = {double(newShape.width), double(newShape.height)};
    double location[2], size[2];
    for (int d = 0; d < 2; ++d) {
        double ca = (double(a[size_t(d)]) + a[size_t(d) + 2]) / (2 * os[d]), cb = (double(b[size_t(d)]) + b[size_t(d) + 2]) / (2 * ns[d]);
        double ea = (double(a[size_t(d) + 2]) - a[size_t(d)]) / os[d], eb = (double(b[size_t(d) + 2]) - b[size_t(d)]) / ns[d];
        location[d] = ca - cb;
        size[d] = std::fabs(std::log(ea / eb));
    }
    double distance = std::sqrt(location[0] * location[0] + location[1] * location[1]);
    return 1.6 * distance + .2 * ((size[0] + size[1]) / 2);
}

std::vector<Pair> pairRegions(const std::vector<Proposal>& oldRegions, const std::vector<Proposal>& newRegions, Shape oldShape,
                              Shape newShape, const Points& source, const Points& target) {
    size_t n = oldRegions.size(), m = newRegions.size();
    if (!n || !m) return {};
    std::vector<std::vector<Evidence>> records(n, std::vector<Evidence>(m));
    std::vector<std::vector<double>> geometry(n, std::vector<double>(m)), costs(n, std::vector<double>(m));
    std::vector<std::vector<bool>> supported(n, std::vector<bool>(m));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) records[i][j] = evidence(oldRegions[i].box, newRegions[j].box, source, target);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) {
            geometry[i][j] = geometryCost(oldRegions[i].box, newRegions[j].box, oldShape, newShape);
            supported[i][j] = records[i][j].score >= 2.5;
            costs[i][j] = geometry[i][j];
            // Match evidence chooses among independently detected contours; unsupported pairs keep the geometry path.
            if (supported[i][j]) costs[i][j] = std::min(costs[i][j], .2 - .025 * std::min(records[i][j].score, 20.));
        }
    // Partial assignment with a dummy unmatched option (cost .62), exact DP over used-new-region bitmasks.
    std::map<std::pair<size_t, uint32_t>, std::pair<double, std::vector<int>>> memo;
    std::function<std::pair<double, std::vector<int>>(size_t, uint32_t)> assign = [&](size_t index, uint32_t used) {
        if (index == n) return std::pair<double, std::vector<int>>{0., {}};
        auto key = std::make_pair(index, used);
        if (auto it = memo.find(key); it != memo.end()) return it->second;
        auto [tail, path] = assign(index + 1, used);
        std::pair<double, std::vector<int>> best{.62 + tail, {-1}};
        best.second.insert(best.second.end(), path.begin(), path.end());
        for (size_t other = 0; other < m; ++other) {
            if ((used & (1u << other)) || costs[index][other] >= .62) continue;
            auto [t, p] = assign(index + 1, used | (1u << other));
            double candidate = costs[index][other] + t;
            if (candidate < best.first) {
                best.first = candidate;
                best.second = {int(other)};
                best.second.insert(best.second.end(), p.begin(), p.end());
            }
        }
        memo[key] = best;
        return best;
    };
    std::vector<int> assignment = assign(0, 0).second;
    std::vector<Pair> accepted;
    for (size_t index = 0; index < n; ++index) {
        int other = assignment[index];
        if (other < 0) continue;
        double alternative = INFINITY, reverse = INFINITY;
        for (size_t j = 0; j < m; ++j)
            if (int(j) != other) alternative = std::min(alternative, costs[index][j]);
        for (size_t i = 0; i < n; ++i)
            if (i != index) reverse = std::min(reverse, costs[i][size_t(other)]);
        double margin = std::min(m > 1 ? alternative : 1., n > 1 ? reverse : 1.) - costs[index][size_t(other)];
        if (margin < .035) continue;  // ambiguous pair
        Pair p;
        p.name = "F" + std::to_string(index) + "-" + std::to_string(other);
        p.oldIndex = int(index);
        p.newIndex = other;
        p.oldBox = oldRegions[index].box;
        p.newBox = newRegions[size_t(other)].box;
        p.cost = pyround(costs[index][size_t(other)], 4);
        p.geometryCost = pyround(geometry[index][size_t(other)], 4);
        p.margin = pyround(margin, 4);
        p.source = supported[index][size_t(other)] ? "direct-match" : "geometry";
        p.evidence = records[index][size_t(other)];
        accepted.push_back(p);
    }
    return accepted;
}

namespace {

std::array<float, 4> endpoints(const Box& b) { return {float(b[0]), float(b[1]), float(b[2]) - 1.f, float(b[3]) - 1.f}; }

std::array<float, 4> commonBox(const Pair& p) {
    if (p.midpoint) return *p.midpoint;
    auto a = endpoints(p.oldBox), b = endpoints(p.newBox);
    return {(a[0] + b[0]) / 2.f, (a[1] + b[1]) / 2.f, (a[2] + b[2]) / 2.f, (a[3] + b[3]) / 2.f};
}

bool overlapDuring(const std::array<float, 4>& a0, const std::array<float, 4>& a1, const std::array<float, 4>& b0,
                   const std::array<float, 4>& b1) {
    double start = 0., end = 1.;
    const int pairs[2][2] = {{2, 0}, {3, 1}};
    for (auto& rl : pairs) {
        int right = rl[0], left = rl[1];
        const std::array<float, 4>* sets[2][4] = {{&a0, &a1, &b0, &b1}, {&b0, &b1, &a0, &a1}};
        for (auto& s : sets) {
            double initial = double((*s[0])[size_t(right)] - (*s[2])[size_t(left)]) - .05;
            double slope = double((*s[1])[size_t(right)] - (*s[3])[size_t(left)]) - .05 - initial;
            if (std::fabs(slope) < 1e-9) {
                if (initial <= 0) return false;
            } else if (slope > 0) start = std::max(start, -initial / slope);
            else end = std::min(end, -initial / slope);
        }
    }
    return start < end - 1e-8;
}

} // namespace

bool conflicts(const Pair& a, const Pair& b) {
    auto am = commonBox(a), bm = commonBox(b);
    return overlapDuring(endpoints(a.oldBox), am, endpoints(b.oldBox), bm) ||
           overlapDuring(am, endpoints(a.newBox), bm, endpoints(b.newBox));
}

Pair candidateWithMidpoint(const Pair& original, Shape oldShape, Shape newShape, Shape size) {
    Pair item = original;
    Shape fallback{pyroundInt((oldShape.width + newShape.width) / 2.), pyroundInt((oldShape.height + newShape.height) / 2.)};
    std::array<float, 4> midpoint = commonBox(item);
    if (size.width != fallback.width || size.height != fallback.height) {
        float sx = float(double(size.width - 1) / (fallback.width - 1)), sy = float(double(size.height - 1) / (fallback.height - 1));
        midpoint = {midpoint[0] * sx, midpoint[1] * sy, midpoint[2] * sx, midpoint[3] * sy};
    }
    // Preserve edges touching either viewport boundary across the whole path.
    auto eo = endpoints(item.oldBox), en = endpoints(item.newBox);
    const int limits[2][3] = {{oldShape.width - 1, newShape.width - 1, size.width - 1}, {oldShape.height - 1, newShape.height - 1, size.height - 1}};
    for (int axis = 0; axis < 2; ++axis) {
        const int edges[2][4] = {{axis, 0, 0, 0}, {axis + 2, limits[axis][0], limits[axis][1], limits[axis][2]}};
        for (auto& e : edges)
            if (std::fabs(double(eo[size_t(e[0])] - float(e[1]))) < .01 || std::fabs(double(en[size_t(e[0])] - float(e[2]))) < .01)
                midpoint[size_t(e[0])] = float(e[3]);
    }
    item.midpoint = midpoint;
    return item;
}

std::vector<Pair> selectSafeCandidates(const std::vector<Pair>& candidates, Shape oldShape, Shape newShape, std::optional<Shape> size) {
    Shape domain = size.value_or(Shape{pyroundInt((oldShape.width + newShape.width) / 2.), pyroundInt((oldShape.height + newShape.height) / 2.)});
    std::vector<Pair> sorted = candidates;
    std::stable_sort(sorted.begin(), sorted.end(), [](const Pair& a, const Pair& b) {
        return std::min(area(a.oldBox), area(a.newBox)) / (1 + a.cost) > std::min(area(b.oldBox), area(b.newBox)) / (1 + b.cost);
    });
    std::vector<Pair> accepted;
    for (auto& original : sorted) {
        Pair item = candidateWithMidpoint(original, oldShape, newShape, domain);
        if (std::none_of(accepted.begin(), accepted.end(), [&](const Pair& other) { return conflicts(item, other); }))
            accepted.push_back(item);
    }
    return accepted;
}

std::vector<Pair> chooseSupported(const std::vector<Proposal>& oldSmall, const std::vector<Proposal>& newSmall,
                                  const Points& source, const Points& target, Shape oldShape, Shape newShape) {
    std::vector<Pair> candidates;
    for (size_t i = 0; i < oldSmall.size(); ++i)
        for (size_t j = 0; j < newSmall.size(); ++j) {
            Evidence proof = evidence(oldSmall[i].box, newSmall[j].box, source, target);
            if (proof.score < 5) continue;
            Pair p;
            p.name = "S" + std::to_string(i) + "-" + std::to_string(j);
            p.oldIndex = int(i);
            p.newIndex = int(j);
            p.oldBox = oldSmall[i].box;
            p.newBox = newSmall[j].box;
            p.cost = std::max(0., .35 - .01 * proof.score);
            p.evidence = proof;
            candidates.push_back(p);
        }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Pair& a, const Pair& b) { return a.evidence.score > b.evidence.score; });
    auto ratio = [](const Box& a, const Box& b) { return double(intersection(a, b)) / std::max(1, std::min(area(a), area(b))); };
    std::vector<Pair> chosen;
    for (auto& item : candidates)
        if (std::none_of(chosen.begin(), chosen.end(), [&](const Pair& o) {
                return ratio(item.oldBox, o.oldBox) > .15 || ratio(item.newBox, o.newBox) > .15;
            }))
            chosen.push_back(item);
    return selectSafeCandidates(chosen, oldShape, newShape);
}

bool overlapsCoarse(const Pair& pair, const std::vector<Pair>& coarse) {
    for (auto& item : coarse)
        for (int side = 0; side < 2; ++side)
            if (double(intersection(pair.box(side), item.box(side))) / std::max(1, area(pair.box(side))) > .15) return true;
    return false;
}

bool containedPair(const Pair& pair, const Leaf& leaf) {
    for (int side = 0; side < 2; ++side) {
        const Box& l = leaf.box(side);
        const Box& p = pair.box(side);
        if (!(l[0] <= p[0] && l[1] <= p[1] && p[2] <= l[2] && p[3] <= l[3])) return false;
    }
    return true;
}

void verifiedLandmarks(const Points& source, const Points& target, const std::vector<Pair>& pairs, Points& trustedSource,
                       Points& trustedTarget) {
    std::vector<uint8_t> chosen(source.size(), 0);
    for (auto& pair : pairs) {
        Points a, b;
        std::vector<int> index;
        inside(pair.oldBox, pair.newBox, source, target, a, b, &index);
        if (a.size() < 6) continue;
        cv::Mat mask;
        cv::estimateAffinePartial2D(a, b, mask, cv::RANSAC, 2.5);
        if (mask.empty() || cv::countNonZero(mask) < 6) continue;
        for (size_t k = 0; k < index.size(); ++k)
            if (mask.at<uint8_t>(int(k))) chosen[size_t(index[k])] = 1;
    }
    trustedSource.clear();
    trustedTarget.clear();
    for (size_t i = 0; i < source.size(); ++i)
        if (chosen[i]) {
            trustedSource.push_back(source[i]);
            trustedTarget.push_back(target[i]);
        }
}

} // namespace prep
