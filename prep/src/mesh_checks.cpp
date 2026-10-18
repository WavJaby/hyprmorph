#include "mesh_checks.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace prep {

cv::Mat crop(const cv::Mat& image, const Box& box) {
    return image(cv::Range(box[1], box[3]), cv::Range(box[0], box[2]));
}

double minimumPathArea(const Grid& from, const Grid& to) {
    auto f = [](double v) { return double(float(v)); };
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r + 1 < from.rows; ++r)
        for (int c = 0; c + 1 < from.cols; ++c) {
            // Triangles (p, q, s) and (p, s, r) of the cell, p = (r, c), q = (r, c+1), r' = (r+1, c), s = (r+1, c+1).
            const int tri[2][3][2] = {{{r, c}, {r, c + 1}, {r + 1, c + 1}}, {{r, c}, {r + 1, c + 1}, {r + 1, c}}};
            for (auto& t : tri) {
                double a[3][2], b[3][2];
                for (int k = 0; k < 3; ++k)
                    for (int d = 0; d < 2; ++d) {
                        a[k][d] = f(from.at(t[k][0], t[k][1], d));
                        b[k][d] = f(to.at(t[k][0], t[k][1], d));
                    }
                double u0 = a[1][0] - a[0][0], u1 = a[1][1] - a[0][1];
                double v0 = a[2][0] - a[0][0], v1 = a[2][1] - a[0][1];
                double du0 = (b[1][0] - b[0][0]) - u0, du1 = (b[1][1] - b[0][1]) - u1;
                double dv0 = (b[2][0] - b[0][0]) - v0, dv1 = (b[2][1] - b[0][1]) - v1;
                double c0 = u0 * v1 - u1 * v0;
                double c1 = (du0 * v1 - du1 * v0) + (u0 * dv1 - u1 * dv0);
                double c2 = du0 * dv1 - du1 * dv0;
                double minimum = std::min(c0, c0 + c1 + c2);
                if (c2 > 0) {
                    double time = -c1 / (2 * c2);
                    if (time > 0 && time < 1) {
                        double stationary = c0 + c1 * time + c2 * (time * time);
                        if (stationary < minimum) minimum = stationary;
                    }
                }
                best = std::min(best, minimum / 2);
            }
        }
    return best;
}

bool Protection::any() const {
    return std::any_of(cell.begin(), cell.end(), [](uint8_t v) { return v != 0; });
}

Protection deformationGroups(const cv::Mat& patch, const Grid& source, bool flatOnly) {
    Protection out;
    int rows = source.rows, cols = source.cols;
    out.rows = rows - 1;
    out.cols = cols - 1;
    out.cell.assign(size_t(out.rows) * size_t(out.cols), 0);
    if (flatOnly)
        for (int row = 0; row < rows - 1; ++row)
            for (int column = 0; column < cols - 1; ++column) {
                double ax = source.x(row, column), ay = source.y(row, column);
                double bx = source.x(row + 1, column + 1), by = source.y(row + 1, column + 1);
                int lo0 = std::max(int(std::floor(ax)) - 2, 0), lo1 = std::max(int(std::floor(ay)) - 2, 0);
                int hi0 = std::min(int(std::ceil(bx)) + 3, patch.cols), hi1 = std::min(int(std::ceil(by)) + 3, patch.rows);
                int lo[3] = {255, 255, 255}, hi[3] = {0, 0, 0};
                for (int y = lo1; y < hi1; ++y) {
                    const uint8_t* p = patch.ptr<uint8_t>(y);
                    for (int x = lo0; x < hi0; ++x)
                        for (int ch = 0; ch < 3; ++ch) {
                            lo[ch] = std::min<int>(lo[ch], p[x * 3 + ch]);
                            hi[ch] = std::max<int>(hi[ch], p[x * 3 + ch]);
                        }
                }
                int range = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
                out.cell[size_t(row) * size_t(out.cols) + size_t(column)] = range > 12;
            }
    for (int dimension = 0; dimension < 2; ++dimension) {
        std::vector<int> parent(size_t(rows) * size_t(cols));
        for (size_t i = 0; i < parent.size(); ++i) parent[i] = int(i);
        auto root = [&](int index) {
            while (parent[size_t(index)] != index) {
                parent[size_t(index)] = parent[size_t(parent[size_t(index)])];
                index = parent[size_t(index)];
            }
            return index;
        };
        for (int row = 0; row < out.rows; ++row)
            for (int column = 0; column < out.cols; ++column) {
                if (!out.cell[size_t(row) * size_t(out.cols) + size_t(column)]) continue;
                int a = row * cols + column, b = a + 1, c = a + cols, d = a + cols + 1;
                int edges[2][2] = {{a, c}, {b, d}};
                if (dimension == 1) {
                    edges[0][0] = a; edges[0][1] = b;
                    edges[1][0] = c; edges[1][1] = d;
                }
                for (auto& e : edges) {
                    int second = root(e[1]);
                    parent[size_t(second)] = root(e[0]);
                }
            }
        std::vector<int> roots(parent.size());
        for (size_t i = 0; i < parent.size(); ++i) roots[i] = root(int(i));
        std::vector<int> unique = roots;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        auto& labels = out.groups[dimension];
        labels.resize(roots.size());
        for (size_t i = 0; i < roots.size(); ++i)
            labels[i] = int(std::lower_bound(unique.begin(), unique.end(), roots[i]) - unique.begin());
    }
    return out;
}

double rectangleError(const Grid& grid, const Protection& protection) {
    float worst = 0;
    bool any = false;
    for (int r = 0; r < protection.rows; ++r)
        for (int c = 0; c < protection.cols; ++c) {
            if (!protection.cell[size_t(r) * size_t(protection.cols) + size_t(c)]) continue;
            auto g = [&](int rr, int cc, int d) { return float(grid.at(rr, cc, d)); };
            float values[4] = {std::fabs(g(r, c, 0) - g(r + 1, c, 0)), std::fabs(g(r, c + 1, 0) - g(r + 1, c + 1, 0)),
                               std::fabs(g(r, c, 1) - g(r, c + 1, 1)), std::fabs(g(r + 1, c, 1) - g(r + 1, c + 1, 1))};
            for (float v : values) worst = any ? std::max(worst, v) : v, any = true;
        }
    return any ? double(worst) : 0.;
}

bool auditLeaf(const cv::Mat& oldImage, const cv::Mat& newImage, const Leaf& leaf) {
    for (int side = 0; side < 2; ++side) {
        const Box& box = leaf.box(side);
        if (box[2] - box[0] <= 0 || box[3] - box[1] <= 0) return false;
        if (!leaf.meshes) continue;
        const Grid& source = leaf.meshes->source(side);
        const Grid& midpoint = leaf.meshes->midpoint(side);
        if (!(minimumPathArea(source, midpoint) > .01)) return false;
        Grid source32 = source;
        for (double& v : source32.xy) v = double(float(v));
        Protection protection = deformationGroups(crop(side ? newImage : oldImage, box), source32, true);
        if (!(rectangleError(midpoint, protection) < .001)) return false;
    }
    return true;
}

} // namespace prep
