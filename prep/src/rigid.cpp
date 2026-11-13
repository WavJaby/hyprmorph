#include "rigid.hpp"

#include "mesh_checks.hpp"
#include "numeric.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace prep {
namespace {

// Tuned thresholds of the strip matcher (analysis px / NCC units).
constexpr double ACCEPT = .93, MARGIN = .035;
constexpr int REVERSE_TOLERANCE = 4, MAX_GROWN_WIDTH = 48, FOREGROUND_STEP = 18;

// Per column of rows [y, y + height): max over rows/channels of |v - median over rows| > 18 (complete_foreground mask).
std::vector<uint8_t> foregroundMask(const cv::Mat& image, int y, int height) {
    std::vector<uint8_t> mask(size_t(image.cols), 0);
    std::vector<float> values(static_cast<size_t>(height));
    for (int x = 0; x < image.cols; ++x)
        for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < height; ++r) values[size_t(r)] = image.ptr<uint8_t>(y + r)[x * 3 + c];
            float median = numeric::median(values);
            for (int r = 0; r < height; ++r)
                if (std::fabs(float(image.ptr<uint8_t>(y + r)[x * 3 + c]) - median) > FOREGROUND_STEP) mask[size_t(x)] = 1;
        }
    return mask;
}

// Exact NCC of two equal-size patches (integer sums), at least as exact as the float32 NCC the thresholds were tuned on.
double patchNcc(const cv::Mat& a, const cv::Mat& b) {
    int64_t n = int64_t(a.rows) * a.cols * 3, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (int y = 0; y < a.rows; ++y) {
        const uint8_t* p = a.ptr<uint8_t>(y);
        const uint8_t* q = b.ptr<uint8_t>(y);
        for (int k = 0; k < a.cols * 3; ++k) {
            int64_t u = p[k], v = q[k];
            sa += u; sb += v; saa += u * u; sbb += v * v; sab += u * v;
        }
    }
    double num = double(n * sab - sa * sb);
    double den = std::sqrt(double(n * saa - sa * sa)) * std::sqrt(double(n * sbb - sb * sb));
    return den > 0 ? num / den : 0.;
}

std::vector<Landmark> completeForeground(const cv::Mat& a, const cv::Mat& b, const std::vector<Landmark>& candidates) {
    std::vector<Landmark> out;
    if (candidates.empty()) return out;
    int height = candidates[0].height;
    std::map<int, std::vector<uint8_t>> masksA, masksB;
    for (const Landmark& item : candidates) {
        int width = item.width;
        int oldX = int(std::nearbyint(item.oldX - (width - 1) / 2.)), newX = int(std::nearbyint(item.newX - (width - 1) / 2.));
        int oldY = int(std::nearbyint(item.oldY)), newY = int(std::nearbyint(item.newY));
        if (!masksA.count(oldY)) masksA[oldY] = foregroundMask(a, oldY, height);
        if (!masksB.count(newY)) masksB[newY] = foregroundMask(b, newY, height);
        int left = 0, right = width - 1;
        for (int side = 0; side < 2; ++side) {
            const std::vector<uint8_t>& mask = side ? masksB[newY] : masksA[oldY];
            int x = side ? newX : oldX, lo = x, hi = x + width - 1, length = int(mask.size());
            while (lo > 2 && mask[size_t(lo - 1)]) --lo;
            while (hi < length - 3 && mask[size_t(hi + 1)]) ++hi;
            left = std::min(left, lo - x - 1);
            right = std::max(right, hi - x + 1);
        }
        if (right - left + 1 > MAX_GROWN_WIDTH || oldX + left < 2 || newX + left < 2 || oldX + right >= a.cols - 2 ||
            newX + right >= b.cols - 2)
            continue;
        int full = right - left + 1;
        cv::Mat pa = a(cv::Range(oldY, oldY + height), cv::Range(oldX + left, oldX + right + 1));
        cv::Mat pb = b(cv::Range(newY, newY + height), cv::Range(newX + left, newX + right + 1));
        double score = patchNcc(pa, pb);
        if (score < ACCEPT) continue;
        Landmark grown = item;
        grown.seedOldX = item.oldX;
        grown.seedNewX = item.newX;
        grown.oldX = oldX + (left + right) / 2.;
        grown.newX = newX + (left + right) / 2.;
        grown.width = full;
        grown.seedWidth = width;
        grown.grown = true;
        grown.score = score;
        out.push_back(grown);
    }
    return out;
}

} // namespace

json::Value Landmark::toJson() const {
    json::Value out = json::Value::object();
    out.set("old_x", oldX);
    out.set("new_x", newX);
    out.set("old_y", oldY);
    out.set("new_y", newY);
    out.set("width", width);
    out.set("height", height);
    out.set("score", score);
    if (grown) {
        out.set("seed_old_x", seedOldX);
        out.set("seed_new_x", seedNewX);
        out.set("seed_width", seedWidth);
        out.set("complete_foreground", true);
    }
    return out;
}

std::vector<Landmark> stripLandmarks(const cv::Mat& a, const cv::Mat& b, const StripScores& scores) {
    if (scores.height < 5) return {};
    std::vector<Landmark> candidates;
    // Patches arrive grouped by width; `reverse` indexes this width's valid-patch list (the reverse match's patch, x = `back`).
    for (size_t begin = 0; begin < scores.patches.size();) {
        size_t end = begin;
        while (end < scores.patches.size() && scores.patches[end].width == scores.patches[begin].width) ++end;
        for (size_t i = begin; i < end; ++i) {
            const PatchScore& p = scores.patches[i];
            if (p.peak < ACCEPT || double(p.peak - p.second) < MARGIN) continue;
            int back = scores.patches[begin + size_t(p.reverse)].x;
            if (std::abs(back - p.x) > REVERSE_TOLERANCE) continue;
            Landmark l;
            l.oldX = p.x + (p.width - 1) / 2.;
            l.newX = p.bestx + (p.width - 1) / 2.;
            l.oldY = scores.oldY;
            l.newY = p.besty;
            l.width = p.width;
            l.height = scores.height;
            l.score = double(p.peak);
            candidates.push_back(l);
        }
        begin = end;
    }
    if (candidates.empty()) return {};
    // Coherent vertical displacement, then the best non-overlapping monotone chain.
    std::vector<double> shifts;
    for (auto& c : candidates) shifts.push_back(c.newY - c.oldY);
    double shift = numeric::median(shifts);
    std::vector<Landmark> kept;
    for (size_t i = 0; i < candidates.size(); ++i)
        if (std::fabs(shifts[i] - shift) <= .5) kept.push_back(candidates[i]);
    std::stable_sort(kept.begin(), kept.end(), [](const Landmark& x, const Landmark& y) { return x.oldX < y.oldX; });
    std::vector<std::pair<double, std::vector<int>>> paths;
    for (size_t i = 0; i < kept.size(); ++i) {
        double weight = kept[i].score * std::sqrt(double(kept[i].width));
        std::pair<double, std::vector<int>> best{weight, {int(i)}};
        for (size_t j = 0; j < i; ++j) {
            double gap = (kept[i].width + kept[j].width) / 2. + 2;
            if (kept[i].oldX - kept[j].oldX >= gap && kept[i].newX - kept[j].newX >= gap) {
                auto& [score, path] = paths[j];
                if (score + weight > best.first) {
                    best.first = score + weight;
                    best.second = path;
                    best.second.push_back(int(i));
                }
            }
        }
        paths.push_back(std::move(best));
    }
    // max() over (score, path) tuples: first maximum, ties broken by lexicographic path.
    size_t top = 0;
    for (size_t i = 1; i < paths.size(); ++i)
        if (paths[i] > paths[top]) top = i;
    std::vector<Landmark> selected;
    for (int i : paths[top].second) selected.push_back(kept[size_t(i)]);
    // Expand after selecting the safe ordered anchors; an expansion must not swallow a neighbouring anchor.
    std::vector<Landmark> grown = completeForeground(a, b, selected);
    std::vector<Landmark> result = selected;
    for (auto& r : result)
        for (auto& g : grown)
            if (g.seedOldX == r.oldX && g.seedNewX == r.newX) { r = g; break; }
    for (size_t pass = 0; pass < result.size(); ++pass) {
        std::set<size_t> bad;
        for (size_t i = 0; i + 1 < result.size(); ++i)
            for (int side = 0; side < 2; ++side) {
                double right = (side ? result[i].newX : result[i].oldX) + (result[i].width - 1) / 2.;
                double left = (side ? result[i + 1].newX : result[i + 1].oldX) - (result[i + 1].width - 1) / 2.;
                if (right >= left - 1) { bad.insert(i); bad.insert(i + 1); }
            }
        if (bad.empty()) break;
        for (size_t i : bad) result[i] = selected[i];
    }
    return result;
}

namespace {

std::optional<std::vector<double>> fitFreeIntervals(const std::vector<double>& axis, const std::vector<bool>& rigid, double last) {
    std::vector<double> steps(axis.size() - 1), fixedSteps, freeSteps;
    for (size_t i = 0; i + 1 < axis.size(); ++i) {
        steps[i] = axis[i + 1] - axis[i];
        (rigid[i] ? fixedSteps : freeSteps).push_back(steps[i]);
    }
    double fixed = numeric::sum(fixedSteps), free = numeric::sum(freeSteps);
    if (free <= 0 || last - fixed <= 0) return std::nullopt;
    std::vector<double> out{axis[0]};
    double total = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
        total += rigid[i] ? steps[i] : steps[i] * (last - fixed) / free;
        out.push_back(axis[0] + total);
    }
    return out;
}

Grid meshgrid(const std::vector<double>& xs, const std::vector<double>& ys, bool vertical) {
    // Horizontal: rows follow ys, columns xs. Vertical strips were transposed: transpose back and swap x/y.
    if (!vertical) {
        Grid g(int(ys.size()), int(xs.size()));
        for (size_t r = 0; r < ys.size(); ++r)
            for (size_t c = 0; c < xs.size(); ++c) {
                g.x(int(r), int(c)) = xs[c];
                g.y(int(r), int(c)) = ys[r];
            }
        return g;
    }
    Grid g(int(xs.size()), int(ys.size()));
    for (size_t r = 0; r < xs.size(); ++r)
        for (size_t c = 0; c < ys.size(); ++c) {
            g.x(int(r), int(c)) = ys[c];
            g.y(int(r), int(c)) = xs[r];
        }
    return g;
}

json::Value rejected(const char* reason) {
    json::Value out = json::Value::object();
    out.set("accepted", false);
    out.set("reason", reason);
    return out;
}

cv::Mat strip(const cv::Mat& patch, bool vertical) {
    if (!vertical) return patch.isContinuous() ? patch : patch.clone();
    cv::Mat t;
    cv::transpose(patch, t);
    return t;
}

} // namespace

StripMesh meshForStrip(const cv::Mat& oldPatch, const cv::Mat& newPatch, bool vertical, const double middle[2],
                       const std::vector<Landmark>& landmarks) {
    StripMesh out;
    cv::Mat a = strip(oldPatch, vertical), b = strip(newPatch, vertical);
    if (landmarks.size() < 2) {
        out.result = rejected("insufficient_unique_rigid_landmarks");
        out.result.set("landmarks", int(landmarks.size()));
        return out;
    }
    double lo = INFINITY, hi = -INFINITY;
    std::vector<double> newYs;
    for (auto& p : landmarks) {
        lo = std::min(lo, p.newY - p.oldY);
        hi = std::max(hi, p.newY - p.oldY);
        newYs.push_back(p.newY);
    }
    if (hi - lo > 1) {
        out.result = rejected("inconsistent_perpendicular_motion");
        return out;
    }
    int height = landmarks[0].height;
    double oldY = landmarks[0].oldY, newY = numeric::median(newYs);
    std::vector<double> axes[2][2];
    for (int side = 0; side < 2; ++side) {
        const cv::Mat& image = side ? b : a;
        double y = side ? newY : oldY;
        std::vector<double> x{0.};
        for (auto& p : landmarks)
            for (double offset : {-(p.width - 1) / 2., (p.width - 1) / 2.}) x.push_back((side ? p.newX : p.oldX) + offset);
        x.push_back(double(image.cols - 1));
        std::vector<double> ys{0., y, y + height - 1, double(image.rows - 1)};
        for (auto* axis : {&x, &ys})
            for (size_t i = 0; i + 1 < axis->size(); ++i)
                if ((*axis)[i + 1] - (*axis)[i] <= 0) {
                    out.result = rejected("nonpositive_interval");
                    return out;
                }
        axes[side][0] = x;
        axes[side][1] = ys;
    }
    std::vector<double> mid[2];
    for (int i = 0; i < 2; ++i)
        for (size_t k = 0; k < axes[0][i].size(); ++k) mid[i].push_back((axes[0][i][k] + axes[1][i][k]) / 2);
    if (middle) {
        double last[2] = {(vertical ? middle[1] : middle[0]) - 1, (vertical ? middle[0] : middle[1]) - 1};
        std::vector<bool> rigidX(mid[0].size() - 1), rigidY{false, true, false};
        for (size_t i = 0; i < rigidX.size(); ++i) rigidX[i] = i % 2 == 1;
        for (int i = 0; i < 2; ++i) {
            if (numeric::isclose(mid[i].back(), last[i])) continue;
            auto fitted = fitFreeIntervals(mid[i], i ? rigidY : rigidX, last[i]);
            if (!fitted) {
                out.result = rejected("midpoint_extent_below_rigid_size");
                return out;
            }
            mid[i] = *fitted;
        }
    }
    Meshes m;
    m.oldSource = meshgrid(axes[0][0], axes[0][1], vertical);
    m.oldMidpoint = meshgrid(mid[0], mid[1], vertical);
    m.newSource = meshgrid(axes[1][0], axes[1][1], vertical);
    m.newMidpoint = meshgrid(mid[0], mid[1], vertical);
    out.mesh = std::move(m);
    for (size_t index = 0; index < landmarks.size(); ++index) {
        int i = int(index);
        std::array<int, 4> window{2 * i + 1, 1, 2 * i + 2, 2};
        if (vertical) window = {1, 2 * i + 1, 2, 2 * i + 2};
        out.windows.push_back(window);
    }
    double minimum = INFINITY;
    for (auto& side : axes)
        for (auto& axis : side)
            for (size_t i = 0; i + 1 < axis.size(); ++i) minimum = std::min(minimum, axis[i + 1] - axis[i]);
    out.result = json::Value::object();
    out.result.set("accepted", true);
    out.result.set("orientation", vertical ? "vertical" : "horizontal");
    out.result.set("landmarks", int(landmarks.size()));
    json::Array patches;
    for (auto& l : landmarks) patches.push_back(l.toJson());
    out.result.set("matched_patches", patches);
    out.result.set("minimum_interval", minimum);
    return out;
}

StripOrientation stripOrientation(const Leaf& leaf) {
    bool horizontal = true, vertical = true;
    for (int side = 0; side < 2; ++side) {
        const Box& b = leaf.box(side);
        double w = b[2] - b[0], h = b[3] - b[1];
        horizontal &= w / h >= 4;
        vertical &= h / w >= 4;
    }
    return horizontal ? StripOrientation::Horizontal : vertical ? StripOrientation::Vertical : StripOrientation::None;
}

void RigidStage::prepare(const cv::Mat& oldImage, const cv::Mat& newImage, const std::vector<Leaf>& leaves) {
    meshes.clear();
    std::vector<int> index;
    std::vector<std::pair<cv::Mat, cv::Mat>> jobs;
    for (size_t i = 0; i < leaves.size(); ++i) {
        StripOrientation o = stripOrientation(leaves[i]);
        if (o == StripOrientation::None) continue;
        bool vertical = o == StripOrientation::Vertical;
        index.push_back(int(i));
        jobs.emplace_back(strip(crop(oldImage, leaves[i].oldBox), vertical), strip(crop(newImage, leaves[i].newBox), vertical));
    }
    std::vector<StripScores> scores = ncc.score(jobs);
    for (size_t k = 0; k < jobs.size(); ++k) {
        const Leaf& leaf = leaves[size_t(index[k])];
        bool vertical = stripOrientation(leaf) == StripOrientation::Vertical;
        double middle[2] = {leaf.midpointBox[2] - leaf.midpointBox[0], leaf.midpointBox[3] - leaf.midpointBox[1]};
        std::vector<Landmark> landmarks = stripLandmarks(jobs[k].first, jobs[k].second, scores[k]);
        meshes[index[k]] = meshForStrip(crop(oldImage, leaf.oldBox), crop(newImage, leaf.newBox), vertical, middle, landmarks);
    }
}

int RigidStage::strips(std::vector<Leaf>& leaves) {
    int accepted = 0;
    for (auto& [i, strip] : meshes) {
        Leaf& leaf = leaves[size_t(i)];
        if (leaf.meshes) continue;
        json::Value row = json::Value::object();
        row.set("stage", "rigid_strips");
        row.set("region", i);
        row.set("name", leaf.name);
        for (auto& [k, v] : strip.result.obj()) row.set(k, v);
        report.push(row);
        if (!strip.mesh) continue;
        leaf.meshes = strip.mesh;
        leaf.rigid = strip.result;
        leaf.windows = strip.windows;
        ++accepted;
    }
    return accepted;
}

int RigidStage::patches(const cv::Mat& oldImage, const cv::Mat& newImage, std::vector<Leaf>& leaves) {
    int accepted = 0;
    for (auto& [i, strip] : meshes) {
        Leaf& leaf = leaves[size_t(i)];
        json::Value row = json::Value::object();
        row.set("stage", "gpu_patches");
        row.set("region", i);
        for (auto& [k, v] : strip.result.obj()) row.set(k, v);
        if (strip.mesh) {
            Leaf trial = leaf;
            trial.meshes = strip.mesh;
            trial.rigid = strip.result;
            trial.windows = strip.windows;
            // A mesh replacement keeps every box: only the trial leaf's own checks can fail.
            if (auditLeaf(oldImage, newImage, trial)) {
                leaf = std::move(trial);
                ++accepted;
            } else {
                row.set("accepted", false);
                row.set("reason", "safety_audit_rejected");
            }
        }
        report.push(row);
    }
    return accepted;
}

} // namespace prep
