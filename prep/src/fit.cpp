#include "fit.hpp"

#include "mesh_checks.hpp"
#include "numeric.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace prep {
namespace {

// Spacing of regular grid lines and number of extra structural lines per axis (source_grid's structural_lines call).
constexpr int SPACING = 24, MAX_PEAKS = 4;
constexpr int MAX_ITERATIONS = 1500;
constexpr float TOLERANCE = .0005f;

struct GridF {
    int rows = 0, cols = 0;
    std::vector<float> v;  // rows x cols x 2
    GridF() = default;
    GridF(int r, int c) : rows(r), cols(c), v(size_t(r) * size_t(c) * 2, 0.f) {}
    float& at(int r, int c, int d) { return v[(size_t(r) * size_t(cols) + size_t(c)) * 2 + size_t(d)]; }
    float at(int r, int c, int d) const { return v[(size_t(r) * size_t(cols) + size_t(c)) * 2 + size_t(d)]; }
    Grid toGrid() const {
        Grid g(rows, cols);
        for (size_t i = 0; i < v.size(); ++i) g.xy[i] = double(v[i]);
        return g;
    }
};

// shared_mesh_midpoint.structural_lines(image, axis, spacing=24, maximum=4, smooth): regular lines + strongest
// gradient lines, float32 like the reference (column means reduce row by row; row means reduce pairwise).
std::vector<float> structuralLines(const cv::Mat& smooth, int axis) {
    int height = smooth.rows, width = smooth.cols, length = axis ? height : width;
    std::vector<int> base;
    for (int v = 0; v < length; v += SPACING) base.push_back(v);
    base.push_back(length - 1);
    auto gradient = [&](int y0, int x0, int y1, int x1) {
        const float* a = smooth.ptr<float>(y0) + x0 * 3;
        const float* b = smooth.ptr<float>(y1) + x1 * 3;
        float s = 0.f;
        for (int c = 0; c < 3; ++c) {
            float d = b[c] - a[c];
            s += d * d;
        }
        return std::sqrt(s);
    };
    std::vector<float> strength(size_t(length - 1));
    if (axis == 0) {
        std::vector<float> total(size_t(width - 1), 0.f);
        for (int y = 0; y < height; ++y)
            for (int k = 0; k + 1 < width; ++k) total[size_t(k)] += gradient(y, k, y, k + 1);
        for (int k = 0; k + 1 < width; ++k) strength[size_t(k)] = total[size_t(k)] / float(height);
    } else {
        std::vector<float> row(static_cast<size_t>(width));
        for (int k = 0; k + 1 < height; ++k) {
            for (int x = 0; x < width; ++x) row[size_t(x)] = gradient(k, x, k + 1, x);
            strength[size_t(k)] = numeric::pairwise(row.data(), row.size()) / float(width);
        }
    }
    // argsort(strength)[::-1]; equal strengths: higher index first (a stable ascending sort reversed).
    std::vector<int> order(strength.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return strength[size_t(a)] < strength[size_t(b)]; });
    std::reverse(order.begin(), order.end());
    std::vector<int> peaks;
    for (int index : order) {
        int coordinate = index + 1;
        if (coordinate < 10 || coordinate > length - 11) continue;
        if (std::any_of(peaks.begin(), peaks.end(), [&](int p) { return std::abs(coordinate - p) < 12; })) continue;
        if (std::any_of(base.begin(), base.end(), [&](int b) { return std::abs(coordinate - b) < 8; })) continue;
        peaks.push_back(coordinate);
        if (int(peaks.size()) >= MAX_PEAKS) break;
    }
    std::set<int> all(base.begin(), base.end());
    all.insert(peaks.begin(), peaks.end());
    return std::vector<float>(all.begin(), all.end());
}

GridF sourceGrid(const cv::Mat& image, const std::vector<std::array<float, 4>>& boxes) {
    cv::Mat lab, smooth;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32FC3);
    cv::GaussianBlur(lab, smooth, cv::Size(0, 0), 3);
    std::vector<float> axes[2];
    for (int d = 0; d < 2; ++d) {
        int length = d ? image.rows : image.cols;
        std::set<double> mandatory{0., double(length - 1)};
        for (auto& b : boxes) {
            mandatory.insert(double(b[size_t(d)]));
            mandatory.insert(double(b[size_t(d) + 2]));
        }
        std::vector<double> positions(mandatory.begin(), mandatory.end());
        for (float value : structuralLines(smooth, d))
            if (std::all_of(positions.begin(), positions.end(), [&](double e) { return std::fabs(value - e) >= 4; }))
                positions.push_back(double(value));
        std::sort(positions.begin(), positions.end());
        for (double p : positions) axes[d].push_back(float(p));
    }
    GridF g(int(axes[1].size()), int(axes[0].size()));
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c < g.cols; ++c) {
            g.at(r, c, 0) = axes[0][size_t(c)];
            g.at(r, c, 1) = axes[1][size_t(r)];
        }
    return g;
}

struct Problem {
    GridF source, baseline, displacement, pinned;
    std::vector<uint8_t> fixed;  // rows x cols x 2
    Protection protection;
    std::vector<uint8_t> locked[2];
    std::vector<double> pins[2];
    std::vector<float> horizontal, vertical, count;
};

std::array<float, 4> endpoints(const Box& b) { return {float(b[0]), float(b[1]), float(b[2]) - 1.f, float(b[3]) - 1.f}; }

// prepare_side: nullopt = rejected (incompatible shared constraints / protected rectangle conflict).
std::optional<Problem> prepareSide(const cv::Mat& image, Shape size, const std::vector<Pair>& pairs, int side) {
    std::vector<std::array<float, 4>> boxes;
    for (auto& p : pairs) {
        auto e = endpoints(p.box(side));
        boxes.push_back({e[0], e[1], e[2], e[3]});
    }
    Problem pr;
    pr.source = sourceGrid(image, boxes);
    int R = pr.source.rows, C = pr.source.cols;
    float scale[2] = {float(double(size.width - 1) / (image.cols - 1)), float(double(size.height - 1) / (image.rows - 1))};
    pr.baseline = pr.source;
    for (size_t i = 0; i < pr.baseline.v.size(); ++i) pr.baseline.v[i] *= scale[i % 2];
    pr.displacement = GridF(R, C);
    pr.fixed.assign(size_t(R) * size_t(C) * 2, 0);
    auto fixedAt = [&](int r, int c, int d) -> uint8_t& { return pr.fixed[(size_t(r) * size_t(C) + size_t(c)) * 2 + size_t(d)]; };
    for (int r = 0; r < R; ++r) fixedAt(r, 0, 0) = fixedAt(r, C - 1, 0) = 1;
    for (int c = 0; c < C; ++c) fixedAt(0, c, 1) = fixedAt(R - 1, c, 1) = 1;
    for (auto& item : pairs) {
        auto box = endpoints(item.box(side));
        const auto& destination = *item.midpoint;
        std::vector<std::pair<int, int>> inside;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                float x = pr.source.at(r, c, 0), y = pr.source.at(r, c, 1);
                if (x >= box[0] && y >= box[1] && x <= box[2] && y <= box[3]) inside.emplace_back(r, c);
            }
        std::vector<float> values;
        for (auto [r, c] : inside)
            for (int d = 0; d < 2; ++d) {
                float target = destination[size_t(d)] + (pr.source.at(r, c, d) - box[size_t(d)]) / (box[size_t(d) + 2] - box[size_t(d)]) *
                                                            (destination[size_t(d) + 2] - destination[size_t(d)]);
                float value = target - pr.baseline.at(r, c, d);
                if (fixedAt(r, c, d) && std::fabs(pr.displacement.at(r, c, d) - value) > .02f) return std::nullopt;
                values.push_back(value);
            }
        size_t k = 0;
        for (auto [r, c] : inside)
            for (int d = 0; d < 2; ++d) {
                pr.displacement.at(r, c, d) = values[k++];
                fixedAt(r, c, d) = 1;
            }
    }
    pr.pinned = pr.displacement;
    Grid source64 = pr.source.toGrid();
    pr.protection = deformationGroups(image, source64, true);
    for (int d = 0; d < 2; ++d) {
        const auto& group = pr.protection.groups[d];
        int n = *std::max_element(group.begin(), group.end()) + 1;
        std::vector<double> lo(size_t(n), INFINITY), hi(size_t(n), -INFINITY);
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c) {
                if (!fixedAt(r, c, d)) continue;
                size_t g = size_t(group[size_t(r * C + c)]);
                double v = pr.displacement.at(r, c, d);
                lo[g] = std::min(lo[g], v);
                hi[g] = std::max(hi[g], v);
            }
        for (int g = 0; g < n; ++g)
            if (hi[size_t(g)] - lo[size_t(g)] > .02) return std::nullopt;
        pr.locked[d].resize(size_t(n));
        pr.pins[d].resize(size_t(n));
        for (int g = 0; g < n; ++g) {
            pr.locked[d][size_t(g)] = std::isfinite(lo[size_t(g)]);
            pr.pins[d][size_t(g)] = std::isfinite(lo[size_t(g)]) ? lo[size_t(g)] : 0.;
        }
    }
    pr.horizontal.resize(size_t(C - 1));
    pr.vertical.resize(size_t(R - 1));
    for (int c = 0; c + 1 < C; ++c) pr.horizontal[size_t(c)] = 1.f / std::max(pr.source.at(0, c + 1, 0) - pr.source.at(0, c, 0), .01f);
    for (int r = 0; r + 1 < R; ++r) pr.vertical[size_t(r)] = 1.f / std::max(pr.source.at(r + 1, 0, 1) - pr.source.at(r, 0, 1), .01f);
    pr.count.assign(size_t(R) * size_t(C), 0.f);
    auto count = [&](int r, int c) -> float& { return pr.count[size_t(r) * size_t(C) + size_t(c)]; };
    for (int r = 0; r < R; ++r)
        for (int c = 1; c < C; ++c) count(r, c) += pr.horizontal[size_t(c - 1)];
    for (int r = 0; r < R; ++r)
        for (int c = 0; c + 1 < C; ++c) count(r, c) += pr.horizontal[size_t(c)];
    for (int r = 1; r < R; ++r)
        for (int c = 0; c < C; ++c) count(r, c) += pr.vertical[size_t(r - 1)];
    for (int r = 0; r + 1 < R; ++r)
        for (int c = 0; c < C; ++c) count(r, c) += pr.vertical[size_t(r)];
    return pr;
}

// structure_midpoint.solve_reference: weighted Jacobi + group projection, float32 grid / float64 group sums.
GridF solveReference(const Problem& pr) {
    int R = pr.source.rows, C = pr.source.cols;
    GridF displacement = pr.displacement, neighbors(R, C), updated(R, C);
    std::vector<double> totals, weights;
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        std::fill(neighbors.v.begin(), neighbors.v.end(), 0.f);
        for (int r = 1; r < R; ++r)
            for (int c = 0; c < C; ++c)
                for (int d = 0; d < 2; ++d) neighbors.at(r, c, d) += displacement.at(r - 1, c, d) * pr.vertical[size_t(r - 1)];
        for (int r = 0; r + 1 < R; ++r)
            for (int c = 0; c < C; ++c)
                for (int d = 0; d < 2; ++d) neighbors.at(r, c, d) += displacement.at(r + 1, c, d) * pr.vertical[size_t(r)];
        for (int r = 0; r < R; ++r)
            for (int c = 1; c < C; ++c)
                for (int d = 0; d < 2; ++d) neighbors.at(r, c, d) += displacement.at(r, c - 1, d) * pr.horizontal[size_t(c - 1)];
        for (int r = 0; r < R; ++r)
            for (int c = 0; c + 1 < C; ++c)
                for (int d = 0; d < 2; ++d) neighbors.at(r, c, d) += displacement.at(r, c + 1, d) * pr.horizontal[size_t(c)];
        for (size_t i = 0; i < updated.v.size(); ++i)
            updated.v[i] = pr.fixed[i] ? pr.pinned.v[i] : neighbors.v[i] / pr.count[i / 2];
        for (int d = 0; d < 2; ++d) {
            const auto& group = pr.protection.groups[d];
            size_t n = pr.locked[d].size();
            totals.assign(n, 0.);
            weights.assign(n, 0.);
            for (size_t i = 0; i < group.size(); ++i) {
                totals[size_t(group[i])] += double(updated.v[i * 2 + size_t(d)] * pr.count[i]);
                weights[size_t(group[i])] += double(pr.count[i]);
            }
            for (size_t g = 0; g < n; ++g) totals[g] = pr.locked[d][g] ? pr.pins[d][g] : totals[g] / weights[g];
            for (size_t i = 0; i < group.size(); ++i) updated.v[i * 2 + size_t(d)] = float(totals[size_t(group[i])]);
        }
        float change = 0.f;
        for (size_t i = 0; i < updated.v.size(); ++i) change = std::max(change, std::fabs(updated.v[i] - displacement.v[i]));
        std::swap(displacement, updated);
        if (change < TOLERANCE) break;
    }
    return displacement;
}

// finish_side: target mesh, or nullopt on a fold along the half path / boundary order violation.
std::optional<std::pair<GridF, GridF>> finishSide(const Problem& pr, const GridF& displacement) {
    GridF target = pr.baseline;
    for (size_t i = 0; i < target.v.size(); ++i) target.v[i] += displacement.v[i];
    if (!(minimumPathArea(pr.source.toGrid(), target.toGrid()) > .01)) return std::nullopt;
    int R = target.rows, C = target.cols;
    for (int c = 0; c + 1 < C; ++c)
        if (target.at(0, c + 1, 0) - target.at(0, c, 0) <= 0 || target.at(R - 1, c + 1, 0) - target.at(R - 1, c, 0) <= 0) return std::nullopt;
    for (int r = 0; r + 1 < R; ++r)
        if (target.at(r + 1, 0, 1) - target.at(r, 0, 1) <= 0 || target.at(r + 1, C - 1, 1) - target.at(r, C - 1, 1) <= 0) return std::nullopt;
    return std::make_pair(pr.source, target);
}

std::optional<std::pair<GridF, GridF>> fitSide(const cv::Mat& image, Shape size, const std::vector<Pair>& pairs, int side) {
    auto problem = prepareSide(image, size, pairs, side);
    if (!problem) return std::nullopt;
    return finishSide(*problem, solveReference(*problem));
}

} // namespace

std::vector<RefineReport> refineRegions(const cv::Mat& oldImage, const cv::Mat& newImage, std::vector<Leaf>& leaves,
                                        const std::vector<Pair>& candidates) {
    std::vector<RefineReport> results;
    for (auto& leaf : leaves) {
        if (leaf.kind == "top-band") continue;
        std::vector<Pair> local;
        for (auto& item : candidates) {
            bool inside = true;
            for (int side = 0; side < 2 && inside; ++side) {
                const Box& b = item.box(side);
                const Box& p = leaf.box(side);
                inside = b[0] >= p[0] && b[1] >= p[1] && b[2] <= p[2] && b[3] <= p[3];
            }
            if (!inside) continue;
            Pair l;
            l.name = item.name;
            l.cost = item.cost;
            for (int k = 0; k < 4; ++k) {
                l.oldBox[size_t(k)] = item.oldBox[size_t(k)] - leaf.oldBox[size_t(k % 2)];
                l.newBox[size_t(k)] = item.newBox[size_t(k)] - leaf.newBox[size_t(k % 2)];
            }
            local.push_back(l);
        }
        if (local.empty()) continue;
        cv::Mat patches[2] = {crop(oldImage, leaf.oldBox), crop(newImage, leaf.newBox)};
        Shape shapes[2] = {{patches[0].cols, patches[0].rows}, {patches[1].cols, patches[1].rows}};
        Shape size{pyroundInt(leaf.midpointBox[2] - leaf.midpointBox[0]), pyroundInt(leaf.midpointBox[3] - leaf.midpointBox[1])};
        // Larger contours first. A rejection keeps the complete texture rather than treating uncertainty as an exit.
        std::stable_sort(local.begin(), local.end(), [](const Pair& a, const Pair& b) {
            return std::min(area(a.oldBox), area(a.newBox)) / (1 + a.cost) > std::min(area(b.oldBox), area(b.newBox)) / (1 + b.cost);
        });
        std::vector<Pair> accepted;
        std::optional<std::array<std::pair<GridF, GridF>, 2>> last;
        for (auto& original : local) {
            Pair item = candidateWithMidpoint(original, shapes[0], shapes[1], size);
            if (std::any_of(accepted.begin(), accepted.end(), [&](const Pair& o) { return conflicts(item, o); })) continue;
            std::vector<Pair> trial = accepted;
            trial.push_back(item);
            auto a = fitSide(patches[0], size, trial, 0);
            if (!a) continue;
            auto b = fitSide(patches[1], size, trial, 1);
            if (!b) continue;
            accepted.push_back(item);
            last = std::array<std::pair<GridF, GridF>, 2>{*a, *b};
        }
        RefineReport report{leaf.name, {}};
        for (auto& p : accepted) report.accepted.push_back(p.name);
        results.push_back(report);
        // The last accepted trial already solved exactly the accepted constraints. Nothing accepted: no local mesh.
        if (accepted.empty()) continue;
        Meshes m;
        m.oldSource = (*last)[0].first.toGrid();
        m.oldMidpoint = (*last)[0].second.toGrid();
        m.newSource = (*last)[1].first.toGrid();
        m.newMidpoint = (*last)[1].second.toGrid();
        leaf.meshes = std::move(m);
    }
    return results;
}

} // namespace prep
