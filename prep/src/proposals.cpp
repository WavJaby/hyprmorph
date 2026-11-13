#include "proposals.hpp"

#include <opencv2/features.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <map>

namespace prep {

int area(const Box& b) { return std::max(0, b[2] - b[0]) * std::max(0, b[3] - b[1]); }

int intersection(const Box& a, const Box& b) {
    return area({std::max(a[0], b[0]), std::max(a[1], b[1]), std::min(a[2], b[2]), std::min(a[3], b[3])});
}

double iou(const Box& a, const Box& b) {
    int overlap = intersection(a, b);
    return double(overlap) / double(std::max(1, area(a) + area(b) - overlap));
}

double pyround(double value, int digits) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.*f", digits, value);
    return std::strtod(buffer, nullptr);
}

json::Value Proposal::toJson() const {
    json::Value out = json::Value::object();
    out.set("box", json::Array{box[0], box[1], box[2], box[3]});
    out.set("boundary_score", score);
    out.set("method", method);
    return out;
}

namespace {

// Hamming distances of every 32-byte ORB descriptor pair (rows of a x rows of b), hardware popcount when available.
[[gnu::always_inline]] inline std::vector<uint16_t> hammingWith(const cv::Mat& a, const cv::Mat& b) {
    CV_Assert(a.type() == CV_8U && b.type() == CV_8U && a.cols == 32 && b.cols == 32);
    std::vector<uint16_t> out(size_t(a.rows) * size_t(b.rows));
    std::vector<uint64_t> wa(size_t(a.rows) * 4), wb(size_t(b.rows) * 4);
    for (int i = 0; i < a.rows; ++i) std::memcpy(&wa[size_t(i) * 4], a.ptr(i), 32);
    for (int j = 0; j < b.rows; ++j) std::memcpy(&wb[size_t(j) * 4], b.ptr(j), 32);
    for (int i = 0; i < a.rows; ++i) {
        const uint64_t* p = &wa[size_t(i) * 4];
        uint16_t* row = &out[size_t(i) * size_t(b.rows)];
        for (int j = 0; j < b.rows; ++j) {
            const uint64_t* q = &wb[size_t(j) * 4];
            row[j] = uint16_t(__builtin_popcountll(p[0] ^ q[0]) + __builtin_popcountll(p[1] ^ q[1]) + __builtin_popcountll(p[2] ^ q[2]) +
                              __builtin_popcountll(p[3] ^ q[3]));
        }
    }
    return out;
}

// Same body compiled twice; always_inline makes __builtin_popcountll expand under the caller's target.
__attribute__((target("popcnt"))) std::vector<uint16_t> hammingPopcnt(const cv::Mat& a, const cv::Mat& b) { return hammingWith(a, b); }
std::vector<uint16_t> hammingGeneric(const cv::Mat& a, const cv::Mat& b) { return hammingWith(a, b); }

std::vector<uint16_t> hammingTable(const cv::Mat& a, const cv::Mat& b) {
    return __builtin_cpu_supports("popcnt") ? hammingPopcnt(a, b) : hammingGeneric(a, b);
}

} // namespace

void matchImages(const cv::Mat& oldImage, const cv::Mat& newImage, Points& source, Points& target) {
    source.clear();
    target.clear();
    auto detect = [](const cv::Mat& image, std::vector<cv::KeyPoint>& keys, cv::Mat& descriptors) {
        auto detector = cv::ORB::create(1500, 1.2f, 8, 15, 0, 2, cv::ORB::HARRIS_SCORE, 31, 10);
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        detector->detectAndCompute(gray, cv::noArray(), keys, descriptors);
    };
    cv::Mat descA, descB;
    std::vector<cv::KeyPoint> keysA, keysB;
    // The two detections are independent (own detector each): the old image runs on a second thread.
    auto other = std::async(std::launch::async, [&] { detect(oldImage, keysA, descA); });
    detect(newImage, keysB, descB);
    other.get();
    if (descA.empty() || descB.empty()) return;
    // BFMatcher.knnMatch(k=2) both ways, from one shared Hamming distance table. A pair survives only with a unique
    // nearest neighbour (ratio .8 fails on a tie), so the matcher's tie order cannot change the result.
    int n = descA.rows, m = descB.rows;
    if (n < 2 || m < 2) return;
    std::vector<uint16_t> distance = hammingTable(descA, descB);
    // One row-major pass keeps the two nearest per row (forward) and per column (backward); strict < keeps the
    // lowest index among equal distances, as a scan in index order does.
    struct Nearest { int first = -1, d1 = 1 << 20, d2 = 1 << 20; };
    auto offer = [](Nearest& best, int d, int index) {
        if (d < best.d1) { best.d2 = best.d1; best.d1 = d; best.first = index; }
        else if (d < best.d2) best.d2 = d;
    };
    std::vector<Nearest> rows(static_cast<size_t>(n)), columns(static_cast<size_t>(m));
    for (int i = 0; i < n; ++i) {
        const uint16_t* row = &distance[size_t(i) * size_t(m)];
        for (int j = 0; j < m; ++j) {
            offer(rows[size_t(i)], row[j], j);
            offer(columns[size_t(j)], row[j], i);
        }
    }
    for (int i = 0; i < n; ++i) {
        const Nearest& f = rows[size_t(i)];
        if (!(f.d1 < .8 * f.d2)) continue;
        const Nearest& b = columns[size_t(f.first)];
        if (!(b.d1 < .8 * b.d2) || b.first != i) continue;
        source.push_back(keysA[size_t(i)].pt);
        target.push_back(keysB[size_t(f.first)].pt);
    }
}

namespace {

// Contours of both Gaussian/Canny passes (sigma .7 then 1.5), each closed with a 3x3 kernel; shared by large/small.
std::vector<std::vector<cv::Point>> contourPasses(const cv::Mat& gray) {
    std::vector<std::vector<cv::Point>> all;
    for (double sigma : {0.7, 1.5}) {
        cv::Mat smooth, edges;
        cv::GaussianBlur(gray, smooth, cv::Size(0, 0), sigma);
        cv::Canny(smooth, edges, 20, 60);
        cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, cv::Mat::ones(3, 3, CV_8U));
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        all.insert(all.end(), contours.begin(), contours.end());
    }
    return all;
}

std::vector<Proposal> dedupe(std::vector<Proposal> candidates) {
    std::stable_sort(candidates.begin(), candidates.end(), [](const Proposal& a, const Proposal& b) { return a.score > b.score; });
    std::vector<Proposal> out;
    for (auto& c : candidates)
        if (std::none_of(out.begin(), out.end(), [&](const Proposal& o) { return iou(c.box, o.box) > .88; })) out.push_back(c);
    return out;
}

// Large region proposals from contours.
std::vector<Proposal> proposals(const cv::Mat& gray, const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<Proposal> candidates;
    double imageArea = double(gray.rows) * gray.cols;
    for (auto& contour : contours) {
        cv::Rect r = cv::boundingRect(contour);
        if (r.width < 24 || r.height < 24 || r.width * r.height > .55 * imageArea) continue;
        if (double(std::min(r.width, r.height)) / std::max(r.width, r.height) < .15) continue;
        double rectangularity = cv::contourArea(contour) / double(r.width * r.height);
        if (rectangularity < .65) continue;
        candidates.push_back({{r.x, r.y, r.x + r.width, r.y + r.height}, pyround(rectangularity, 3), "closed-contour"});
    }
    // A card cut by the viewport has no closed contour: pair long vertical boundaries reaching the bottom edge.
    cv::Mat blurred, edges, vertical, labels, stats, centroids;
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 0);
    cv::Canny(blurred, edges, 20, 60);
    cv::morphologyEx(edges, vertical, cv::MORPH_CLOSE, cv::Mat::ones(5, 1, CV_8U));
    cv::morphologyEx(vertical, vertical, cv::MORPH_OPEN, cv::Mat::ones(30, 1, CV_8U));
    int count = cv::connectedComponentsWithStats(vertical, labels, stats, centroids);
    struct Line { int x, top; };
    std::vector<Line> lines;
    for (int i = 1; i < count; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT), y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (w <= 3 && h >= 48 && y + h >= gray.rows - 3) lines.push_back({x, y});
    }
    for (size_t i = 0; i < lines.size(); ++i)
        for (size_t j = i + 1; j < lines.size(); ++j) {
            int x0 = std::min(lines[i].x, lines[j].x), x1 = std::max(lines[i].x, lines[j].x);
            int y0 = std::min(lines[i].top, lines[j].top);
            int width = x1 + 1 - x0, height = gray.rows - y0;
            if (std::abs(lines[i].top - lines[j].top) > 6 || width < 24 || width * height > .55 * imageArea) continue;
            int supported = 0;
            for (int x = x0; x <= x1; ++x) {
                bool any = false;
                for (int y = std::max(0, y0 - 3); y < std::min(gray.rows, y0 + 4); ++y) any |= edges.at<uint8_t>(y, x) > 0;
                supported += any;
            }
            if (double(supported) / (x1 - x0 + 1) < .4) continue;
            candidates.push_back({{x0, y0, x1 + 1, gray.rows}, .9, "viewport-clipped"});
        }
    return dedupe(candidates);
}

// Small candidates: near-rectangular contour boxes, sides >= 12, area 200..2000.
std::vector<Proposal> smallProposals(const std::vector<std::vector<cv::Point>>& contours) {
    constexpr int MIN_SIDE = 12, MIN_AREA = 200;
    std::vector<Proposal> candidates;
    for (auto& contour : contours) {
        cv::Rect r = cv::boundingRect(contour);
        int a = r.width * r.height;
        if (r.width < MIN_SIDE || r.height < MIN_SIDE || a < MIN_AREA || a >= 2000 ||
            double(std::min(r.width, r.height)) / std::max(r.width, r.height) < .15)
            continue;
        double rectangularity = cv::contourArea(contour) / a;
        if (rectangularity < .65) continue;
        candidates.push_back({{r.x, r.y, r.x + r.width, r.y + r.height}, pyround(rectangularity, 3), "small-closed-contour"});
    }
    return dedupe(candidates);
}

// Open outlines: a top edge + a long aligned side imply an occluded rectangle.
std::vector<Proposal> openOutlines(const cv::Mat& gray) {
    cv::Mat edges;
    cv::Canny(gray, edges, 20, 60);
    std::vector<cv::Vec4i> detected;
    cv::HoughLinesP(edges, detected, 1, CV_PI / 180, 35, 35, 35);
    struct Segment { int at, lo, hi; };
    std::vector<Segment> vertical, horizontal;
    for (auto& l : detected) {
        int a = l[0], b = l[1], c = l[2], d = l[3];
        if (std::abs(c - a) <= 2 && std::abs(d - b) >= 80) vertical.push_back({int((a + c) / 2.), std::min(b, d), std::max(b, d)});
        if (std::abs(d - b) <= 2 && std::abs(c - a) >= 64) horizontal.push_back({int((b + d) / 2.), std::min(a, c), std::max(a, c)});
    }
    std::vector<Proposal> result;
    for (auto& h : horizontal) {
        if (h.at < std::max(40., .1 * gray.rows)) continue;
        for (auto& v : vertical) {
            if (std::abs(h.at - v.lo) > 5 || std::min(std::abs(v.at - h.lo), std::abs(v.at - h.hi)) > 5) continue;
            Box box{std::min(v.at, h.lo), std::min(h.at, v.lo), std::max(v.at, h.hi) + 1, v.hi + 1};
            if (area(box) > .55 * gray.rows * gray.cols) continue;
            if (std::min(box[2] - box[0], box[3] - box[1]) < 64) continue;
            if (std::any_of(result.begin(), result.end(), [&](const Proposal& p) { return iou(box, p.box) > .88; })) continue;
            result.push_back({box, .85, "open-outline"});
        }
    }
    return result;
}

double centreInside(const Box& outer, const Box& item) {
    double cx = (item[0] + item[2]) / 2., cy = (item[1] + item[3]) / 2.;
    return outer[0] - 4 <= cx && cx <= outer[2] + 4 && outer[1] - 4 <= cy && cy <= outer[3] + 4;
}

// discover(image, image): proposals(original) + proposals(filtered) are the same list twice here,
// and the duplicate copy can never be selected (it overlaps its own twin), so one copy is used.
std::vector<Proposal> discover(const cv::Mat& gray, const std::vector<Proposal>& found) {
    std::vector<Proposal> outlines = openOutlines(gray), candidates = found;
    std::vector<Proposal> kept;
    // An inferred open box must not slice through an observed closed object outside its own extent.
    for (auto& o : outlines) {
        bool slices = std::any_of(candidates.begin(), candidates.end(), [&](const Proposal& item) {
            double ratio = double(intersection(o.box, item.box)) / area(item.box);
            return item.method == "closed-contour" && area(item.box) >= 2000 && .10 < ratio && ratio < .80 && !centreInside(o.box, item.box);
        });
        if (!slices) kept.push_back(o);
    }
    outlines.swap(kept);
    kept.clear();
    // Open-edge inference must not supersede an observed closed/clipped box with substantially different extents.
    for (auto& o : outlines) {
        bool superseded = std::any_of(candidates.begin(), candidates.end(), [&](const Proposal& item) {
            return area(item.box) >= 2000 && item.score >= .83 &&
                   (iou(item.box, o.box) > .5 || double(intersection(item.box, o.box)) / area(item.box) > .75) &&
                   (item.method == "viewport-clipped" || std::abs(item.box[0] - o.box[0]) > 5 || std::abs(item.box[2] - o.box[2]) > 5 ||
                    std::abs(item.box[1] - o.box[1]) > 32 || std::abs(item.box[3] - o.box[3]) > 10);
        });
        if (!superseded) kept.push_back(o);
    }
    outlines.swap(kept);
    // An outer photo edge is a more precise hypothesis than a contour that includes its title header.
    std::vector<Proposal> merged;
    for (auto& item : candidates)
        if (std::none_of(outlines.begin(), outlines.end(), [&](const Proposal& o) {
                return iou(item.box, o.box) > .7 && std::abs(item.box[0] - o.box[0]) <= 5 && std::abs(item.box[2] - o.box[2]) <= 5;
            }))
            merged.push_back(item);
    merged.insert(merged.end(), outlines.begin(), outlines.end());
    std::stable_sort(merged.begin(), merged.end(), [](const Proposal& a, const Proposal& b) { return area(a.box) > area(b.box); });
    std::vector<Proposal> selected;
    for (auto& item : merged) {
        if (area(item.box) < 2000) continue;
        if (std::any_of(selected.begin(), selected.end(), [&](const Proposal& o) {
                return double(intersection(item.box, o.box)) / std::min(area(item.box), area(o.box)) > .15;
            }))
            continue;
        selected.push_back(item);
    }
    if (selected.size() > 12) selected.resize(12);
    return selected;
}

} // namespace

ImageProposals propose(const cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    auto contours = contourPasses(gray);
    ImageProposals out;
    out.regions = discover(gray, proposals(gray, contours));
    out.small = smallProposals(contours);
    return out;
}

} // namespace prep
