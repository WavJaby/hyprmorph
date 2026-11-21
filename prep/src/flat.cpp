#include "flat.hpp"

#include "mesh_checks.hpp"
#include "numeric.hpp"
#include "pairing.hpp"

#include <opencv2/geometry.hpp>

#include <cmath>
#include <optional>

namespace prep {

std::vector<uint8_t> identicalIntervals(const cv::Mat& patch, int axis) {
    int lines = axis ? patch.rows : patch.cols, across = axis ? patch.cols : patch.rows;
    std::vector<uint8_t> flat(size_t(lines - 1));
    for (int k = 0; k + 1 < lines; ++k) {
        int step = 0;
        for (int s = 0; s < across && step < IDENTICAL_STEP; ++s) {
            const uint8_t* p = axis ? patch.ptr<uint8_t>(k) + s * 3 : patch.ptr<uint8_t>(s) + k * 3;
            const uint8_t* q = axis ? patch.ptr<uint8_t>(k + 1) + s * 3 : patch.ptr<uint8_t>(s) + (k + 1) * 3;
            for (int c = 0; c < 3; ++c) step = std::max(step, std::abs(int(p[c]) - int(q[c])));
        }
        flat[size_t(k)] = step < IDENTICAL_STEP;
    }
    return flat;
}

std::vector<uint8_t> flatIntervals(std::vector<uint8_t> flat) {
    for (size_t i = 0; i < flat.size();) {
        if (!flat[i]) { ++i; continue; }
        size_t j = i;
        while (j < flat.size() && flat[j]) ++j;
        if (j - i < size_t(MIN_FLAT_RUN)) std::fill(flat.begin() + long(i), flat.begin() + long(j), 0);
        i = j;
    }
    return flat;
}

namespace {

// Tuned constants; the accepted flat-absorb results depend on these exact values.
constexpr int VERTEX_SPACING = 24, MIN_SCALE_EVIDENCE = 8;
constexpr double MIN_FLAT_SCALE = .2;
const double NATIVE_SCALE_TOLERANCE = std::log(1.05);

// Which end of an axis the content stays attached to when the leaf resizes: the end at which the old and new line
// profiles (mean colour of each line) agree better over their common length. Revealed or hidden content sits at the
// other end (a top-anchored list growing shows more items at the bottom). Equal agreement -> start.
bool attachedToEnd(const cv::Mat& a, const cv::Mat& b, int axis) {
    cv::Mat pa, pb;
    // reduce dim 0 collapses rows -> one value per column (axis 0 lines); dim 1 -> one per row (axis 1 lines).
    cv::reduce(a, pa, axis ? 1 : 0, cv::REDUCE_AVG, CV_64F);
    cv::reduce(b, pb, axis ? 1 : 0, cv::REDUCE_AVG, CV_64F);
    pa = pa.reshape(1, int(pa.total()));
    pb = pb.reshape(1, int(pb.total()));
    int la = pa.rows / 3, lb = pb.rows / 3, m = std::min(la, lb);
    double start = 0, end = 0;
    for (int i = 0; i < m * 3; ++i) {
        start += std::fabs(pa.at<double>(i) - pb.at<double>(i));
        end += std::fabs(pa.at<double>((la - m) * 3 + i) - pb.at<double>((lb - m) * 3 + i));
    }
    return end < start;
}

// Isotropic-shrink leftover on an axis with no flat run: the identical-line run nearest the free end (one demoted as an
// inter-glyph gap) becomes the axis's flat span, so the region's own background takes the size change instead of a
// single-axis stretch of the content. All-zero when the axis has no identical lines at all (stays forced).
std::vector<uint8_t> backgroundAbsorber(const std::vector<uint8_t>& identical, bool toEnd) {
    std::vector<uint8_t> flat(identical.size(), 0);
    // Free end = opposite the attached end: scan from it inward for the first run.
    long n = long(identical.size()), k = toEnd ? 0 : n - 1, step = toEnd ? 1 : -1;
    while (k >= 0 && k < n && !identical[size_t(k)]) k += step;
    while (k >= 0 && k < n && identical[size_t(k)]) flat[size_t(k)] = 1, k += step;
    return flat;
}

std::vector<double> isotonic(const std::vector<double>& values, const std::vector<double>& weights) {
    struct Block { double value, weight; size_t count; };
    std::vector<Block> blocks;
    for (size_t i = 0; i < values.size(); ++i) {
        blocks.push_back({values[i], weights[i], 1});
        while (blocks.size() > 1 && blocks[blocks.size() - 2].value > blocks.back().value) {
            Block v = blocks.back();
            blocks.pop_back();
            Block& last = blocks.back();
            // An infinite weight pins the pooled value to that member (both pinned: the conflict averages).
            if (std::isinf(v.weight) || std::isinf(last.weight)) {
                double total = 0;
                int n = 0;
                if (std::isinf(last.weight)) { total += last.value; ++n; }
                if (std::isinf(v.weight)) { total += v.value; ++n; }
                last = {total / n, INFINITY, last.count + v.count};
            } else
                last = {(last.value * last.weight + v.value * v.weight) / (last.weight + v.weight), last.weight + v.weight, last.count + v.count};
        }
    }
    std::vector<double> out;
    for (auto& b : blocks) out.insert(out.end(), b.count, b.value);
    return out;
}

std::pair<double, bool> affordableScale(const std::vector<uint8_t>& flat, double target, double scale) {
    double n = double(flat.size()) + 1;
    double flats = 0;
    for (uint8_t f : flat) flats += f;
    double content = n - 1 - flats, gaps = flats * MIN_FLAT_SCALE;
    if (!content || !flats || gaps >= target) return {target / (n - 1), false};
    return {std::min(scale, (target - gaps) / content), true};
}

void subdivide(const std::vector<double>& source, const std::vector<double>& middle, std::vector<double>& outSource,
               std::vector<double>& outMiddle) {
    outSource = {source[0]};
    outMiddle = {middle[0]};
    for (size_t i = 0; i + 1 < source.size(); ++i) {
        double a = source[i], b = source[i + 1], m0 = middle[i], m1 = middle[i + 1];
        int count = std::max(1, int(std::ceil((b - a) / VERTEX_SPACING)));
        for (int k = 1; k <= count; ++k) {
            double fraction = double(k) / count;
            outSource.push_back(a + (b - a) * fraction);
            outMiddle.push_back(m0 + (m1 - m0) * fraction);
        }
    }
}

// One axis: content spans keep their length (times scale), placed by a weighted isotonic fit; flats absorb the rest.
void axisMap(const std::vector<uint8_t>& flat, double target, const std::vector<std::array<double, 2>>* anchors, double scale,
             std::vector<double>& outSource, std::vector<double>& outMiddle) {
    size_t n = flat.size() + 1;
    std::vector<double> source{0.};
    for (size_t i = 0; i + 1 < flat.size(); ++i)
        if (flat[i + 1] != flat[i]) source.push_back(double(i + 1));
    if (source.back() != double(n - 1)) source.push_back(double(n - 1));
    size_t spans = source.size() - 1;
    std::vector<uint8_t> kinds(spans);
    std::vector<double> lengths(spans), gaps(spans), widths(spans);
    std::vector<size_t> content;
    bool anyFlat = false;
    for (size_t i = 0; i < spans; ++i) {
        kinds[i] = flat[size_t(source[i])];
        lengths[i] = source[i + 1] - source[i];
        gaps[i] = kinds[i] ? lengths[i] * MIN_FLAT_SCALE : 0.;
        widths[i] = kinds[i] ? 0. : lengths[i] * scale;
        if (!kinds[i]) content.push_back(i);
        anyFlat |= kinds[i] != 0;
    }
    double gapSum = numeric::sum(gaps);
    if (content.empty() || !anyFlat || gapSum >= target) {
        // Nothing to keep, nothing to absorb with, or not even the flat floor fits: uniform scale.
        std::vector<double> middle;
        for (double s : source) middle.push_back(s * target / double(n - 1));
        subdivide(source, middle, outSource, outMiddle);
        return;
    }
    double widthSum = numeric::sum(widths);
    if (widthSum + gapSum > target)
        for (double& w : widths) w *= (target - gapSum) / widthSum;
    std::vector<double> required(spans);
    double running = 0;
    for (size_t i = 0; i < spans; ++i) {
        required[i] = running;
        running += widths[i] + gaps[i];
    }
    size_t k = content.size();
    std::vector<double> ideal(k), weights(k, 1.), shifted(k);
    for (size_t j = 0; j < k; ++j) {
        size_t i = content[j];
        ideal[j] = (source[i] + lengths[i] / 2) * target / double(n - 1) - widths[i] / 2;
    }
    if (anchors && !anchors->empty())
        for (size_t j = 0; j < k; ++j) {
            size_t i = content[j];
            double own = widths[i] / lengths[i];
            std::vector<double> inside;
            for (auto& a : *anchors)
                if (a[0] >= source[i] && a[0] <= source[i + 1]) inside.push_back(a[1] - (a[0] - source[i]) * own);
            if (!inside.empty()) {
                ideal[j] = numeric::mean(inside);
                weights[j] = 1 + double(inside.size());
            }
        }
    size_t last = content.back();
    std::vector<double> tail(gaps.begin() + long(last) + 1, gaps.end());
    double upper = target - (required[last] + widths[last] + numeric::sum(tail));
    for (size_t j = 0; j < k; ++j) shifted[j] = ideal[j] - required[content[j]];
    // Content touching a box edge stays on it: the box edge is the leaf's interface with its neighbour.
    if (content[0] == 0) { shifted[0] = 0.; weights[0] = INFINITY; }
    if (last == spans - 1) { shifted[k - 1] = upper; weights[k - 1] = INFINITY; }
    std::vector<double> fitted = isotonic(shifted, weights);
    std::vector<double> middle(source.size());
    middle[0] = 0.;
    middle.back() = target;
    for (size_t j = 0; j < k; ++j) {
        size_t i = content[j];
        double start = std::clamp(fitted[j], 0., upper) + required[i];
        middle[i] = start;
        middle[i + 1] = start + widths[i];
    }
    subdivide(source, middle, outSource, outMiddle);
}

Grid meshgrid(const std::vector<double>& xs, const std::vector<double>& ys) {
    Grid g(int(ys.size()), int(xs.size()));
    for (size_t r = 0; r < ys.size(); ++r)
        for (size_t c = 0; c < xs.size(); ++c) {
            g.x(int(r), int(c)) = xs[c];
            g.y(int(r), int(c)) = ys[r];
        }
    return g;
}

std::optional<double> contentScale(const std::vector<cv::Point2d>& a, const std::vector<cv::Point2d>& b) {
    if (a.size() < size_t(MIN_SCALE_EVIDENCE)) return std::nullopt;
    std::vector<cv::Point2f> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    cv::Mat inliers;
    cv::Mat model = cv::estimateAffinePartial2D(fa, fb, inliers, cv::RANSAC, 2.5);
    if (model.empty() || cv::countNonZero(inliers) < MIN_SCALE_EVIDENCE) return std::nullopt;
    return std::hypot(model.at<double>(0, 0), model.at<double>(1, 0));
}

} // namespace

std::vector<FlatResult> applyFlatAbsorb(const cv::Mat& oldImage, const cv::Mat& newImage, std::vector<Leaf>& leaves,
                                        const Points& source, const Points& target) {
    std::vector<FlatResult> results;
    for (size_t index = 0; index < leaves.size(); ++index) {
        Leaf& leaf = leaves[index];
        if (leaf.meshes) continue;
        const Box& ob = leaf.oldBox;
        const Box& nb = leaf.newBox;
        cv::Mat patches[2] = {crop(oldImage, ob), crop(newImage, nb)};
        // Correspondences are pixel-edge coordinates; meshes use pixel centres.
        std::vector<cv::Point2d> ma, mb;
        for (size_t i = 0; i < source.size(); ++i)
            if (contained(source[i], ob) && contained(target[i], nb)) {
                ma.emplace_back(double(source[i].x) - ob[0] - .5, double(source[i].y) - ob[1] - .5);
                mb.emplace_back(double(target[i].x) - nb[0] - .5, double(target[i].y) - nb[1] - .5);
            }
        std::optional<double> scale = contentScale(ma, mb);
        if (!scale) {
            // No match evidence: an isotropic leaf resize is the signature of a scaled image (UI reflow changes one axis).
            double r0 = std::log(double(nb[2] - nb[0]) / (ob[2] - ob[0])), r1 = std::log(double(nb[3] - nb[1]) / (ob[3] - ob[1]));
            if (std::fabs(r0 - r1) <= NATIVE_SCALE_TOLERANCE) scale = std::exp((r0 + r1) / 2);
        }
        if (!scale || std::fabs(std::log(*scale)) <= NATIVE_SCALE_TOLERANCE) scale = 1.;
        double middle[2] = {leaf.midpointBox[2] - leaf.midpointBox[0], leaf.midpointBox[3] - leaf.midpointBox[1]};
        FlatResult result{int(index), leaf.name, false, pyround(*scale, 3), {}, {}};
        // separable_mesh
        double targets[2] = {middle[0] - 1, middle[1] - 1};
        if (std::min({patches[0].rows, patches[0].cols, patches[1].rows, patches[1].cols}) < 2 || std::min(targets[0], targets[1]) <= 0) {
            results.push_back(result);
            continue;
        }
        std::vector<std::array<double, 2>> agreed;
        if (!ma.empty()) {
            double average[2] = {(double(patches[0].cols) + patches[1].cols) / 2 - 1, (double(patches[0].rows) + patches[1].rows) / 2 - 1};
            for (size_t i = 0; i < ma.size(); ++i)
                agreed.push_back({(ma[i].x + mb[i].x) / 2 * (middle[0] - 1) / average[0], (ma[i].y + mb[i].y) / 2 * (middle[1] - 1) / average[1]});
        }
        double meet = (1 + *scale) / 2, owns[2] = {meet, meet / *scale};
        std::vector<uint8_t> identical[2][2], flats[2][2];
        std::pair<double, bool> fits[2][2];
        for (int side = 0; side < 2; ++side)
            for (int axis = 0; axis < 2; ++axis) {
                identical[side][axis] = identicalIntervals(patches[side], axis);
                flats[side][axis] = flatIntervals(identical[side][axis]);
                fits[side][axis] = affordableScale(flats[side][axis], targets[axis], owns[side]);
            }
        // Content that does not fit shrinks isotropically by the smallest scale an absorbing axis can afford;
        // a forced axis (no flat run) would be uniform whatever the factor and does not drive it.
        auto isotropicScale = [&](int side) {
            double factor = 1.;
            for (auto& [value, absorbs] : fits[side])
                if (absorbs) factor = std::min(factor, value / owns[side]);
            return owns[side] * factor;
        };
        // A forced axis whose uniform scale is visibly off the content's isotropic scale squashes the content: its
        // background run absorbs instead. Repeat until stable -- a promoted axis that must shrink lowers the factor,
        // which can make the other axis visibly anisotropic in turn (at most one promotion per side and axis).
        std::optional<bool> toEnd[2];
        for (int side = 0; side < 2; ++side) {
            bool promoted[2] = {false, false};
            for (bool changed = true; changed;) {
                changed = false;
                double isotropic = isotropicScale(side);
                for (int axis = 0; axis < 2; ++axis) {
                    auto [value, absorbs] = fits[side][axis];
                    if (absorbs || promoted[axis] || std::fabs(std::log(value / isotropic)) <= NATIVE_SCALE_TOLERANCE) continue;
                    promoted[axis] = true;
                    if (!toEnd[axis]) toEnd[axis] = attachedToEnd(patches[0], patches[1], axis);
                    std::vector<uint8_t> background = backgroundAbsorber(identical[side][axis], *toEnd[axis]);
                    if (std::none_of(background.begin(), background.end(), [](uint8_t f) { return f != 0; })) continue;
                    flats[side][axis] = std::move(background);
                    fits[side][axis] = affordableScale(flats[side][axis], targets[axis], owns[side]);
                    result.background.push_back(std::string(side ? "new" : "old") + (axis ? "-y-" : "-x-") + (*toEnd[axis] ? "end" : "start"));
                    changed |= fits[side][axis].second;
                }
            }
        }
        Meshes mesh;
        bool useful = false;
        for (int side = 0; side < 2; ++side) {
            double isotropic = isotropicScale(side);
            bool unabsorbed = false;
            // Residual isotropy violation: a forced axis visibly off the isotropic scale with no background run to absorb.
            for (auto& [value, absorbs] : fits[side]) unabsorbed |= !absorbs && std::fabs(std::log(value / isotropic)) > NATIVE_SCALE_TOLERANCE;
            std::vector<double> axes[2][2];
            for (int axis = 0; axis < 2; ++axis) {
                std::vector<std::array<double, 2>> anchors;
                const std::vector<cv::Point2d>& points = side ? mb : ma;
                for (size_t i = 0; i < agreed.size(); ++i) anchors.push_back({axis ? points[i].y : points[i].x, agreed[i][size_t(axis)]});
                axisMap(flats[side][axis], targets[axis], agreed.empty() ? nullptr : &anchors, isotropic, axes[axis][0], axes[axis][1]);
                bool anyFlat = std::any_of(flats[side][axis].begin(), flats[side][axis].end(), [](uint8_t f) { return f != 0; });
                int extent = axis ? patches[side].rows : patches[side].cols;
                useful |= anyFlat && !numeric::isclose(targets[axis], double(extent - 1));
            }
            mesh.source(side) = meshgrid(axes[0][0], axes[1][0]);
            mesh.midpoint(side) = meshgrid(axes[0][1], axes[1][1]);
            if (unabsorbed) result.unabsorbed.push_back(side ? "new" : "old");
        }
        result.accepted = useful;
        if (useful) leaf.meshes = std::move(mesh);
        results.push_back(result);
    }
    return results;
}

} // namespace prep
