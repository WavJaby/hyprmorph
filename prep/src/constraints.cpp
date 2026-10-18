#include "constraints.hpp"

#include "mesh_checks.hpp"
#include "numeric.hpp"
#include "pairing.hpp"

#include <opencv2/geometry.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace prep {
namespace {

// Reference limits (docs/reference/correspondence/rigid-patch-icons.md, mesh constraint layer).
constexpr size_t MIN_MATCHES = 8;
constexpr double MIN_INLIER_RATIO = .55, CELL = 12, MEAN_LIMIT = 6, HELD_OUT_P90_LIMIT = 12, IMPROVEMENT = .1;
constexpr int HELD_OUT_EVERY = 5, DEFAULT_SPACING = 24;
constexpr double STRENGTHS[] = {1., .75, .5, .25, .125};
constexpr float SMOOTH_WEIGHT = .7f, MATCH_WEIGHT = 4;

struct Basis {
    int index[3];
    float weight[3];
};

// Barycentric weights of each point in the grid's triangulation (paired_point_constraints.basis).
std::vector<Basis> basis(const std::vector<cv::Point2d>& points, const Grid& grid) {
    int C = grid.cols, R = grid.rows;
    std::vector<float> xs(static_cast<size_t>(C)), ys(static_cast<size_t>(R));
    for (int c = 0; c < C; ++c) xs[size_t(c)] = float(grid.x(0, c));
    for (int r = 0; r < R; ++r) ys[size_t(r)] = float(grid.y(r, 0));
    std::vector<Basis> out;
    for (auto& p : points) {
        int c = int(std::upper_bound(xs.begin(), xs.end(), p.x, [](double v, float e) { return v < double(e); }) - xs.begin()) - 1;
        int r = int(std::upper_bound(ys.begin(), ys.end(), p.y, [](double v, float e) { return v < double(e); }) - ys.begin()) - 1;
        c = std::clamp(c, 0, C - 2);
        r = std::clamp(r, 0, R - 2);
        double u = std::clamp((p.x - xs[size_t(c)]) / (double(xs[size_t(c) + 1]) - xs[size_t(c)]), 0., 1.);
        double v = std::clamp((p.y - ys[size_t(r)]) / (double(ys[size_t(r) + 1]) - ys[size_t(r)]), 0., 1.);
        int a = r * C + c, b = a + 1, d = a + C + 1, e = a + C;
        Basis bs;
        if (u >= v) bs = {{a, b, d}, {float(1 - u), float(std::fabs(u - v)), float(std::min(u, v))}};
        else bs = {{a, d, e}, {float(1 - v), float(u), float(v - u)}};
        out.push_back(bs);
    }
    return out;
}

// w @ m.reshape(-1, 2): float32, terms in ascending vertex order.
float apply(const Basis& b, const std::vector<float>& m, int d) {
    int order[3] = {0, 1, 2};
    std::sort(order, order + 3, [&](int x, int y) { return b.index[x] < b.index[y]; });
    float s = 0.f;
    for (int k : order) s += b.weight[k] * m[size_t(b.index[k]) * 2 + size_t(d)];
    return s;
}

// Free coordinate groups after merging rigid windows; column of each vertex (-1 = locked on the box boundary).
std::vector<int> freeGroups(const std::vector<int>& groups, int R, int C, int dimension, const std::vector<std::array<int, 4>>& windows,
                            int& count) {
    int n = *std::max_element(groups.begin(), groups.end()) + 1;
    std::vector<int> parent(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) parent[size_t(i)] = i;
    auto root = [&](int i) {
        while (parent[size_t(i)] != i) {
            parent[size_t(i)] = parent[size_t(parent[size_t(i)])];
            i = parent[size_t(i)];
        }
        return i;
    };
    for (auto& [x0, y0, x1, y1] : windows) {
        std::set<int> ids;
        for (int r = y0; r <= std::min(y1, R - 1); ++r)
            for (int c = x0; c <= std::min(x1, C - 1); ++c) ids.insert(groups[size_t(r * C + c)]);
        int first = *ids.begin();
        for (int id : ids)
            if (id != first) parent[size_t(root(id))] = root(first);
    }
    std::vector<int> rooted(groups.size());
    for (size_t i = 0; i < groups.size(); ++i) rooted[i] = root(groups[i]);
    std::set<int> locked;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c) {
            bool boundary = dimension == 0 ? (c == 0 || c == C - 1) : (r == 0 || r == R - 1);
            if (boundary) locked.insert(rooted[size_t(r * C + c)]);
        }
    std::set<int> unique(rooted.begin(), rooted.end());
    std::map<int, int> mapping;
    for (int g : unique)
        if (!locked.count(g)) mapping[g] = int(mapping.size());
    count = int(mapping.size());
    std::vector<int> column(groups.size(), -1);
    for (size_t i = 0; i < groups.size(); ++i)
        if (auto it = mapping.find(rooted[i]); it != mapping.end()) column[i] = it->second;
    return column;
}

std::vector<float> asFloat(const Grid& g) {
    std::vector<float> out(g.xy.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = float(g.xy[i]);
    return out;
}

Grid fromFloat(const std::vector<float>& v, int rows, int cols) {
    Grid g(rows, cols);
    for (size_t i = 0; i < v.size(); ++i) g.xy[i] = double(v[i]);
    return g;
}

double meanOf(const std::vector<double>& errors, const std::vector<uint8_t>& train, bool wantTrain, bool all) {
    std::vector<double> chosen;
    for (size_t i = 0; i < errors.size(); ++i)
        if (all || bool(train[i]) == wantTrain) chosen.push_back(errors[i]);
    return numeric::mean(chosen);
}

struct Region {
    size_t leaf = 0;
    std::vector<cv::Point2d> a, b;
    std::vector<uint8_t> train;
    Meshes mesh;
    Grid grids[2];
    std::vector<float> middles[2];
    std::vector<Basis> bases[2];
    double scales[2][2] = {};
    std::vector<double> priorError;
    Protection protections[2];
    std::vector<int> columns[2][2];  // [dim][side]
    int counts[2][2] = {};
    size_t system[2] = {};
};

std::vector<double> errors(const Region& g, const std::vector<float> (&m)[2]) {
    std::vector<double> out(g.a.size());
    for (size_t p = 0; p < g.a.size(); ++p) {
        double d[2];
        for (int k = 0; k < 2; ++k)
            d[k] = double(apply(g.bases[0][p], m[0], k)) * g.scales[0][k] - double(apply(g.bases[1][p], m[1], k)) * g.scales[1][k];
        out[p] = std::sqrt(d[0] * d[0] + d[1] * d[1]);
    }
    return out;
}

} // namespace

std::vector<ConstraintResult> applyPointConstraints(const cv::Mat& oldImage, const cv::Mat& newImage, std::vector<Leaf>& leaves,
                                                    const Points& source, const Points& target, CgSolver& solver) {
    std::vector<ConstraintResult> results;
    std::vector<Region> regions;
    std::vector<SparseSystem> systems;
    for (size_t index = 0; index < leaves.size(); ++index) {
        const Leaf& leaf = leaves[index];
        const Box& ob = leaf.oldBox;
        const Box& nb = leaf.newBox;
        ConstraintResult result;
        result.region = int(index);
        std::vector<cv::Point2d> a, b;
        for (size_t i = 0; i < source.size(); ++i)
            if (contained(source[i], ob) && contained(target[i], nb)) {
                a.emplace_back(double(source[i].x) - ob[0], double(source[i].y) - ob[1]);
                b.emplace_back(double(target[i].x) - nb[0], double(target[i].y) - nb[1]);
            }
        if (a.size() < MIN_MATCHES) {
            result.reason = "insufficient_same_region_matches";
            results.push_back(result);
            continue;
        }
        cv::Mat inliers;
        cv::Mat affine = cv::estimateAffine2D(a, b, inliers, cv::RANSAC, 2.5, 2000, .995);
        if (affine.empty() || double(cv::countNonZero(inliers)) / double(a.size()) < MIN_INLIER_RATIO) {
            result.reason = "incoherent_local_affine_consensus";
            results.push_back(result);
            continue;
        }
        // One sample per 12 px source/target cell pair limits clusters.
        Region g;
        g.leaf = index;
        std::set<std::array<long, 4>> seen;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!inliers.at<uint8_t>(int(i))) continue;
            std::array<long, 4> key{long(std::floor(a[i].x / CELL)), long(std::floor(a[i].y / CELL)), long(std::floor(b[i].x / CELL)),
                                    long(std::floor(b[i].y / CELL))};
            if (!seen.insert(key).second) continue;
            g.a.push_back(a[i]);
            g.b.push_back(b[i]);
        }
        if (g.a.size() < MIN_MATCHES) {
            result.reason = "insufficient_spatially_distinct_matches";
            results.push_back(result);
            continue;
        }
        for (size_t i = 0; i < g.a.size(); ++i) g.train.push_back(i % HELD_OUT_EVERY != 0);
        cv::Mat patches[2] = {crop(oldImage, ob), crop(newImage, nb)};
        double middleExtent[2] = {leaf.midpointBox[2] - leaf.midpointBox[0] - 1, leaf.midpointBox[3] - leaf.midpointBox[1] - 1};
        if (leaf.meshes) g.mesh = *leaf.meshes;
        else
            for (int side = 0; side < 2; ++side) {
                int size[2] = {patches[side].cols - 1, patches[side].rows - 1};
                std::vector<float> axes[2];
                for (int d = 0; d < 2; ++d) {
                    std::set<double> values;
                    for (int v = 0; v < size[d]; v += DEFAULT_SPACING) values.insert(v);
                    values.insert(double(size[d]));
                    for (double v : values) axes[d].push_back(float(v));
                }
                Grid src(int(axes[1].size()), int(axes[0].size())), mid(src.rows, src.cols);
                for (int r = 0; r < src.rows; ++r)
                    for (int c = 0; c < src.cols; ++c) {
                        float xy[2] = {axes[0][size_t(c)], axes[1][size_t(r)]};
                        for (int d = 0; d < 2; ++d) {
                            src.at(r, c, d) = double(xy[d]);
                            mid.at(r, c, d) = double(xy[d]) / size[d] * middleExtent[d];
                        }
                    }
                g.mesh.source(side) = src;
                g.mesh.midpoint(side) = mid;
            }
        double extent[2] = {leaf.midpointBox[2] - leaf.midpointBox[0], leaf.midpointBox[3] - leaf.midpointBox[1]};
        std::vector<double> points[2][2];
        for (int side = 0; side < 2; ++side) {
            g.grids[side] = fromFloat(asFloat(g.mesh.source(side)), g.mesh.source(side).rows, g.mesh.source(side).cols);
            g.middles[side] = asFloat(g.mesh.midpoint(side));
            g.bases[side] = basis(side ? g.b : g.a, g.grids[side]);
            const Grid& m = g.mesh.midpoint(side);
            float last[2] = {g.middles[side][size_t(m.cols - 1) * 2], g.middles[side][size_t(m.rows - 1) * size_t(m.cols) * 2 + 1]};
            for (int d = 0; d < 2; ++d) g.scales[side][d] = extent[d] / double(last[d]);
            g.protections[side] = deformationGroups(patches[side], g.grids[side], true);
        }
        g.priorError = errors(g, g.middles);
        for (int dim = 0; dim < 2; ++dim) {
            for (int side = 0; side < 2; ++side)
                g.columns[dim][side] = freeGroups(g.protections[side].groups[dim], g.grids[side].rows, g.grids[side].cols, dim, leaf.windows,
                                                  g.counts[dim][side]);
            SparseSystem s;
            s.cols = g.counts[dim][0] + g.counts[dim][1];
            for (size_t p = 0; p < g.a.size(); ++p) {
                if (!g.train[p]) continue;
                for (int side = 0; side < 2; ++side) {
                    // bases @ projection in float32 (vertices of one group sum first), then the float64 scale.
                    std::map<int, float> entries;
                    const Basis& bs = g.bases[side][p];
                    int order[3] = {0, 1, 2};
                    std::sort(order, order + 3, [&](int x, int y) { return bs.index[x] < bs.index[y]; });
                    for (int k : order) {
                        int col = g.columns[dim][side][size_t(bs.index[k])];
                        if (col >= 0) entries[col] += bs.weight[k];
                    }
                    for (auto& [col, value] : entries)
                        s.add(col + (side ? g.counts[dim][0] : 0), (side ? -double(value) : double(value)) * g.scales[side][dim] * MATCH_WEIGHT);
                }
                double rhs = (double(apply(g.bases[1][p], g.middles[1], dim)) * g.scales[1][dim] -
                              double(apply(g.bases[0][p], g.middles[0], dim)) * g.scales[0][dim]) * MATCH_WEIGHT;
                s.endRow(rhs);
            }
            for (int side = 0; side < 2; ++side) {
                int R = g.grids[side].rows, C = g.grids[side].cols, offset = side ? g.counts[dim][0] : 0;
                auto edge = [&](int v0, int v1) {
                    int c0 = g.columns[dim][side][size_t(v0)], c1 = g.columns[dim][side][size_t(v1)];
                    if (c0 == c1) return;  // same free group or both locked: an all-zero row
                    if (c0 >= 0) s.add(c0 + offset, double(SMOOTH_WEIGHT));
                    if (c1 >= 0) s.add(c1 + offset, -double(SMOOTH_WEIGHT));
                    s.endRow(0.);
                };
                for (int r = 0; r < R; ++r)
                    for (int c = 0; c + 1 < C; ++c) edge(r * C + c, r * C + c + 1);
                for (int r = 0; r + 1 < R; ++r)
                    for (int c = 0; c < C; ++c) edge(r * C + c, (r + 1) * C + c);
            }
            g.system[dim] = systems.size();
            systems.push_back(std::move(s));
        }
        results.push_back(result);
        regions.push_back(std::move(g));
    }
    std::vector<std::vector<float>> solutions = solver.solve(systems);
    for (Region& g : regions) {
        ConstraintResult& result = *std::find_if(results.begin(), results.end(), [&](const ConstraintResult& r) { return r.region == int(g.leaf); });
        std::vector<float> corrections[2] = {std::vector<float>(g.middles[0].size(), 0.f), std::vector<float>(g.middles[1].size(), 0.f)};
        for (int dim = 0; dim < 2; ++dim) {
            const std::vector<float>& x = solutions[g.system[dim]];
            for (int side = 0; side < 2; ++side)
                for (size_t v = 0; v < g.columns[dim][side].size(); ++v) {
                    int col = g.columns[dim][side][v];
                    if (col >= 0) corrections[side][v * 2 + size_t(dim)] = x[size_t(col + (side ? g.counts[dim][0] : 0))];
                }
        }
        result.beforeMean = numeric::mean(g.priorError);
        double priorHeld = meanOf(g.priorError, g.train, false, false);
        bool done = false;
        for (double strength : STRENGTHS) {
            std::vector<float> trial[2];
            bool safe = true;
            for (int side = 0; side < 2; ++side) {
                trial[side].resize(g.middles[side].size());
                for (size_t i = 0; i < trial[side].size(); ++i) trial[side][i] = g.middles[side][i] + corrections[side][i] * float(strength);
                int R = g.grids[side].rows, C = g.grids[side].cols;
                Grid m = fromFloat(trial[side], R, C);
                safe &= minimumPathArea(g.grids[side], m) > .01 && rectangleError(m, g.protections[side]) < .001;
                auto at = [&](int r, int c, int d) { return trial[side][(size_t(r) * size_t(C) + size_t(c)) * 2 + size_t(d)]; };
                for (int c = 0; c + 1 < C; ++c) safe &= at(0, c + 1, 0) - at(0, c, 0) > 0 && at(R - 1, c + 1, 0) - at(R - 1, c, 0) > 0;
                for (int r = 0; r + 1 < R; ++r) safe &= at(r + 1, 0, 1) - at(r, 0, 1) > 0 && at(r + 1, C - 1, 1) - at(r, C - 1, 1) > 0;
                float hi[2] = {at(0, C - 1, 0) + .001f, at(R - 1, 0, 1) + .001f};
                for (size_t i = 0; i < trial[side].size(); ++i) safe &= trial[side][i] >= -.001f && trial[side][i] <= hi[i % 2];
            }
            std::vector<double> error = errors(g, trial);
            std::vector<double> held;
            for (size_t i = 0; i < error.size(); ++i)
                if (!g.train[i]) held.push_back(error[i]);
            double mean = numeric::mean(error), heldMean = numeric::mean(held);
            // Held-out evidence must improve as well as the fitted points.
            safe &= mean < result.beforeMean - IMPROVEMENT && heldMean <= priorHeld + IMPROVEMENT;
            safe &= mean <= MEAN_LIMIT && heldMean <= MEAN_LIMIT && numeric::percentile(held, 90) <= HELD_OUT_P90_LIMIT;
            if (!safe) continue;
            Leaf& leaf = leaves[g.leaf];
            g.mesh.oldMidpoint = fromFloat(trial[0], g.mesh.oldMidpoint.rows, g.mesh.oldMidpoint.cols);
            g.mesh.newMidpoint = fromFloat(trial[1], g.mesh.newMidpoint.rows, g.mesh.newMidpoint.cols);
            leaf.meshes = g.mesh;
            result.accepted = true;
            result.strength = strength;
            result.afterMean = mean;
            done = true;
            break;
        }
        if (!done) result.reason = "no_safe_generalizing_update";
    }
    return results;
}

} // namespace prep
