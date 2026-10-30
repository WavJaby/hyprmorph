// Numerics follow the Python reference operation by operation, including where it compares in float32 (numpy array vs Python scalar
// compares at the array's precision) and where in double (values converted with float()). Changing either side breaks decision parity.
#include "readiness.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <map>
#include <sstream>

namespace readiness {
namespace {

// OpenCV BORDER_REFLECT_101 (cv2 default for GaussianBlur and Sobel): -1 -> 1, n -> n-2.
inline int reflect(int i, int n) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
}

// cv2.GaussianBlur(float32, (0,0), sigma): size = round(sigma*4*2+1)|1 for float images; kernel computed in double, stored as float.
std::vector<float> gaussianKernel(double sigma) {
    const int n = static_cast<int>(std::lround(sigma * 8 + 1)) | 1;
    std::vector<float> k(n);
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        const double x = i - (n - 1) * .5;
        k[i] = static_cast<float>(std::exp(-.5 / (sigma * sigma) * x * x));
        sum += k[i];
    }
    for (auto& v : k) v = static_cast<float>(v * (1. / sum));
    return k;
}

// Binary morphology along one axis with OpenCV's rectangular-kernel convention: anchor = length/2, offsets -anchor..length-1-anchor for both
// erode and dilate; pixels outside the image never constrain (cv2 default border value for morphology).
void lineMorph(std::vector<uint8_t>& plane, int width, int height, int length, bool vertical, bool erode, std::vector<uint8_t>& scratch) {
    scratch.assign(plane.size(), 0);
    const int anchor = length / 2;
    if (vertical) {
        // Column prefix counts kept row-major, so every loop runs along contiguous rows.
        thread_local std::vector<int> prefix;
        prefix.assign(static_cast<size_t>(height + 1) * width, 0);
        for (int y = 0; y < height; ++y) {
            const int* above = prefix.data() + static_cast<size_t>(y) * width;
            int* row = prefix.data() + static_cast<size_t>(y + 1) * width;
            const uint8_t* src = plane.data() + static_cast<size_t>(y) * width;
            for (int x = 0; x < width; ++x) row[x] = above[x] + (src[x] != 0);
        }
        for (int y = 0; y < height; ++y) {
            const int lo = std::max(0, y - anchor), hi = std::min(height - 1, y + length - 1 - anchor), span = hi - lo + 1;
            const int* top = prefix.data() + static_cast<size_t>(lo) * width;
            const int* bottom = prefix.data() + static_cast<size_t>(hi + 1) * width;
            uint8_t* dst = scratch.data() + static_cast<size_t>(y) * width;
            if (erode) for (int x = 0; x < width; ++x) dst[x] = bottom[x] - top[x] == span;
            else for (int x = 0; x < width; ++x) dst[x] = bottom[x] - top[x] > 0;
        }
        plane.swap(scratch);
        return;
    }
    const int n = width, lines = height;
    for (int line = 0; line < lines; ++line) {
        auto at = [&](int i) -> size_t { return static_cast<size_t>(line) * width + i; };
        // Prefix count of set pixels along the line: window [lo, hi] is all-set / any-set by count.
        thread_local std::vector<int> prefix;
        prefix.assign(n + 1, 0);
        for (int i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + (plane[at(i)] != 0);
        for (int i = 0; i < n; ++i) {
            const int lo = std::max(0, i - anchor), hi = std::min(n - 1, i + length - 1 - anchor);
            const int set = prefix[hi + 1] - prefix[lo];
            scratch[at(i)] = erode ? set == hi - lo + 1 : set > 0;
        }
    }
    plane.swap(scratch);
}

std::vector<uint8_t> dilateSquare(const std::vector<uint8_t>& plane, int width, int height, int radius) {
    // Separable: OR over the row window, then over the column window; out-of-image pixels never set anything (cv2 dilate border).
    std::vector<uint8_t> rows(plane.size(), 0), out(plane.size(), 0);
    for (int y = 0; y < height; ++y) {
        const uint8_t* src = plane.data() + static_cast<size_t>(y) * width;
        uint8_t* dst = rows.data() + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x)
            if (src[x])
                for (int d = std::max(0, x - radius); d <= std::min(width - 1, x + radius); ++d) dst[d] = 1;
    }
    for (int y = 0; y < height; ++y) {
        uint8_t* dst = out.data() + static_cast<size_t>(y) * width;
        for (int d = std::max(0, y - radius); d <= std::min(height - 1, y + radius); ++d) {
            const uint8_t* src = rows.data() + static_cast<size_t>(d) * width;
            for (int x = 0; x < width; ++x) dst[x] |= src[x];
        }
    }
    return out;
}

double ms(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
} // namespace

int Structure::edgePixels() const {
    int count = 0;
    for (auto v : vertical) count += v != 0;
    for (auto v : horizontal) count += v != 0;
    return count;
}

Color lowPass(const Image& image, const Policy& policy) {
    const int w = image.width, h = image.height;
    const auto kernel = gaussianKernel(policy.blurSigma);
    const int r = static_cast<int>(kernel.size()) / 2, n = static_cast<int>(kernel.size());
    // Horizontal pass reads a reflect-padded copy of each row; vertical pass reads rows through a reflected row table.
    // Summation order (k = -r..r, one product at a time) is the parity contract with the Python reference; keep it.
    std::vector<float> padded(static_cast<size_t>(w + 2 * r) * 3), rows(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        const uint8_t* src = image.data + y * image.stride;
        for (int x = -r; x < w + r; ++x) {
            const uint8_t* pixel = src + (x >= 0 && x < w ? x : reflect(x, w)) * image.channels;
            for (int c = 0; c < 3; ++c) padded[static_cast<size_t>(x + r) * 3 + c] = pixel[c];
        }
        float* dst = rows.data() + static_cast<size_t>(y) * w * 3;
        for (int i = 0; i < w * 3; ++i) dst[i] = 0;
        for (int k = 0; k < n; ++k) {
            const float weight = kernel[k], *src = padded.data() + k * 3;
            for (int i = 0; i < w * 3; ++i) dst[i] += weight * src[i];
        }
    }
    Color out{w, h, std::vector<float>(rows.size())};
    std::vector<const float*> taps(n);
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < n; ++k) taps[k] = rows.data() + static_cast<size_t>(reflect(y + k - r, h)) * w * 3;
        float* dst = out.data.data() + static_cast<size_t>(y) * w * 3;
        for (int i = 0; i < w * 3; ++i) dst[i] = 0;
        for (int k = 0; k < n; ++k) {
            const float weight = kernel[k], *src = taps[k];
            for (int i = 0; i < w * 3; ++i) dst[i] += weight * src[i];
        }
    }
    return out;
}

Structure structure(const Color& color, const Policy& policy) {
    const int w = color.width, h = color.height;
    Structure s{w, h, std::vector<uint8_t>(static_cast<size_t>(w) * h), std::vector<uint8_t>(static_cast<size_t>(w) * h), {}, {}};
    // Reflect-101 padding by one pixel, so the 3x3 Sobel taps need no per-access border logic.
    const size_t pw = static_cast<size_t>(w + 2) * 3;
    std::vector<float> pad(pw * (h + 2));
    for (int y = -1; y <= h; ++y) {
        const float* src = color.data.data() + static_cast<size_t>(reflect(y, h)) * w * 3;
        float* dst = pad.data() + static_cast<size_t>(y + 1) * pw;
        std::copy(src, src + static_cast<size_t>(w) * 3, dst + 3);
        for (int c = 0; c < 3; ++c) {
            dst[c] = src[reflect(-1, w) * 3 + c];
            dst[static_cast<size_t>(w + 1) * 3 + c] = src[reflect(w, w) * 3 + c];
        }
    }
    const float threshold = static_cast<float>(policy.gradientThreshold), dominance = static_cast<float>(policy.gradientDominance);
    std::vector<float> gx(static_cast<size_t>(w) * 3), gy(gx.size());
    for (int y = 0; y < h; ++y) {
        const float *r0 = pad.data() + static_cast<size_t>(y) * pw, *r1 = r0 + pw, *r2 = r1 + pw;
        // cv2.Sobel ksize 3, scale 1/8: derivative [-1 0 1] x smoothing [1 2 1]; i = x*3 + c, tap offsets in padded units.
        for (int i = 0; i < w * 3; ++i) {
            const float dx = (r0[i + 6] - r0[i]) + 2 * (r1[i + 6] - r1[i]) + (r2[i + 6] - r2[i]);
            const float dy = (r2[i] - r0[i]) + 2 * (r2[i + 3] - r0[i + 3]) + (r2[i + 6] - r0[i + 6]);
            gx[i] = std::abs(dx * .125f);
            gy[i] = std::abs(dy * .125f);
        }
        uint8_t *vertical = s.vertical.data() + static_cast<size_t>(y) * w, *horizontal = s.horizontal.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const float mx = std::max(std::max(gx[x * 3], gx[x * 3 + 1]), gx[x * 3 + 2]);
            const float my = std::max(std::max(gy[x * 3], gy[x * 3 + 1]), gy[x * 3 + 2]);
            vertical[x] = mx > threshold && mx > dominance * my;
            horizontal[x] = my > threshold && my > dominance * mx;
        }
    }
    // Python round() is half-to-even; nearbyint in the default rounding mode matches it.
    const int verticalLength = std::max(policy.minimumLinePixels, static_cast<int>(std::nearbyint(h * policy.minimumLineFraction)));
    const int horizontalLength = std::max(policy.minimumLinePixels, static_cast<int>(std::nearbyint(w * policy.minimumLineFraction)));
    std::vector<uint8_t> scratch;
    lineMorph(s.vertical, w, h, verticalLength, true, true, scratch);
    lineMorph(s.vertical, w, h, verticalLength, true, false, scratch);
    lineMorph(s.horizontal, w, h, horizontalLength, false, true, scratch);
    lineMorph(s.horizontal, w, h, horizontalLength, false, false, scratch);
    const int b = policy.borderPixels;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (y < b || y >= h - b || x < b || x >= w - b) s.vertical[static_cast<size_t>(y) * w + x] = s.horizontal[static_cast<size_t>(y) * w + x] = 0;
    s.verticalNear = dilateSquare(s.vertical, w, h, policy.edgeTolerancePixels);
    s.horizontalNear = dilateSquare(s.horizontal, w, h, policy.edgeTolerancePixels);
    return s;
}

Comparison compare(const Structure& a, const Structure& b, const Policy& policy) {
    // Near planes were dilated with the policy that built each Structure; one detector uses one policy, so they agree.
    if (a.width != b.width || a.height != b.height) return {1., 1., 0., 0, 0};
    const int w = a.width, h = a.height, rows = policy.gridRows, columns = policy.gridColumns;
    const int bh = (h + rows - 1) / rows, bw = (w + columns - 1) / columns;
    std::vector<long> support(static_cast<size_t>(rows) * columns, 0), missing(support.size(), 0);
    long totalSupport = 0, totalMissing = 0;
    for (int plane = 0; plane < 2; ++plane) {
        const auto& pa = plane ? a.horizontal : a.vertical;
        const auto& pb = plane ? b.horizontal : b.vertical;
        const auto& da = plane ? a.horizontalNear : a.verticalNear;
        const auto& db = plane ? b.horizontalNear : b.verticalNear;
        for (int y = 0; y < h; ++y) {
            const size_t row = static_cast<size_t>(y) * w;
            long* cellSupport = support.data() + static_cast<size_t>(y / bh) * columns;
            long* cellMissing = missing.data() + static_cast<size_t>(y / bh) * columns;
            for (int x0 = 0, column = 0; x0 < w; x0 += bw, ++column) {
                int s = 0, m = 0;
                for (int x = x0; x < std::min(w, x0 + bw); ++x) {
                    const size_t i = row + x;
                    s += pa[i] + pb[i];
                    m += (pa[i] & (db[i] ^ 1)) + (pb[i] & (da[i] ^ 1));
                }
                cellSupport[column] += s; cellMissing[column] += m;
                totalSupport += s; totalMissing += m;
            }
        }
    }
    Comparison c;
    // float32 division as numpy does it; sums are exact integers in both implementations.
    c.edgeDisagreement = static_cast<float>(totalMissing) / static_cast<float>(std::max(1L, totalSupport));
    const float cellLimit = static_cast<float>(policy.cellDisagreement);
    float worst = 0;
    int stable = 0;
    for (size_t cell = 0; cell < support.size(); ++cell) {
        if (support[cell] < policy.minimumCellSupport) continue;
        ++c.informativeCells;
        const float ratio = static_cast<float>(missing[cell]) / static_cast<float>(support[cell]);
        worst = std::max(worst, ratio);
        if (ratio <= cellLimit) ++stable; else ++c.changingCells;
    }
    c.worstCell = worst;
    c.stableCellFraction = static_cast<double>(stable) / std::max(1, c.informativeCells);
    return c;
}

bool quiet(const Comparison& c, const Policy& policy) {
    if (c.edgeDisagreement > policy.edgeDisagreement) return false;
    return policy.quietRule == QuietRule::WorstTile ? c.worstCell <= policy.cellDisagreement : c.stableCellFraction >= policy.minimumStableCells;
}

AliasDifference aliasDifference(const Color& frame, const Alias& alias, const Policy& policy) {
    if (frame.width != alias.color.width || frame.height != alias.color.height) return {1e9, 1.};  // shape mismatch: certainly not the alias
    double sum = 0;
    size_t count = 0, changed = 0;
    const size_t pixels = static_cast<size_t>(frame.width) * frame.height;
    const float threshold = static_cast<float>(policy.aliasPixelThreshold);
    for (size_t i = 0; i < pixels; ++i) {
        if (!alias.mask.empty() && !alias.mask[i]) continue;
        float delta = 0;
        for (int c = 0; c < 3; ++c) delta = std::max(delta, std::abs(frame.data[i * 3 + c] - alias.color.data[i * 3 + c]));
        sum += delta; ++count; changed += delta > threshold;
    }
    if (!count) return {};
    return {sum / count, static_cast<double>(changed) / count};
}

Detector::Detector(Policy policy, std::vector<Alias> aliases) : policy_(std::move(policy)), aliases_(std::move(aliases)) {}

Observation Detector::observe(const Image& frame, double elapsedMs, const Evidence& evidence) {
    const bool targetGeometry = evidence.targetGeometry;
    const auto started = std::chrono::steady_clock::now();
    Observation o;
    o.elapsedMs = elapsedMs;
    o.targetGeometry = targetGeometry;
    const auto color = lowPass(frame, policy_);
    auto current = structure(color, policy_);
    // Once a new layout painted at the target extent, color returning to an old value does not invalidate that evidence; geometry loss does.
    if (!targetGeometry) updateEvidence_ = false;
    bool reuse = updateEvidence_;
    o.protocolUpdate = policy_.protocolUpdateEvidence && evidence.newBuffer && targetGeometry;
    if (evidence.recordedUpdate) {
        o.updated = *evidence.recordedUpdate || o.protocolUpdate;
        reuse = false;
    } else {
        if (!reuse && !o.protocolUpdate)
            for (const auto& alias : aliases_) o.aliasDifferences.push_back(aliasDifference(color, alias, policy_));
        o.updated = reuse || o.protocolUpdate || (!o.aliasDifferences.empty() && std::all_of(o.aliasDifferences.begin(), o.aliasDifferences.end(), [&](const auto& d) {
            return d.meanDifference >= policy_.updateMeanDifference && d.changedFraction >= policy_.updateChangedFraction;
        }));
    }
    o.updateEvidenceReused = reuse;
    if (targetGeometry && o.updated) updateEvidence_ = true;
    const bool consecutive = !previousMs_ || elapsedMs - *previousMs_ <= policy_.maximumSampleGapMs;
    if (previous_) o.previous = compare(current, *previous_, policy_);
    if (anchor_) o.anchor = compare(current, *anchor_, policy_);
    const bool changed = o.previous && !quiet(*o.previous, policy_);
    o.supported = current.edgePixels() >= policy_.minimumEdgePixels;
    const bool eligible = targetGeometry && o.updated && o.supported;
    const bool layoutMode = policy_.postAcceptance == PostAcceptance::LayoutChange;
    if (layoutMode) {
        // The anchor is the selected frame while ready, so this is the change against what the mesh was built from.
        const bool largeChange = o.anchor && o.anchor->edgeDisagreement >= policy_.revokeEdgeDisagreement;
        o.revoked = readyMs_ && (!eligible || largeChange);
    } else {
        o.revoked = readyMs_ && (!eligible || changed || (o.anchor && !quiet(*o.anchor, policy_)));
    }
    if (o.revoked) readyMs_.reset();
    if (!eligible) {
        anchor_.reset(); quietSinceMs_.reset(); quietSamples_ = 0;
    } else if (layoutMode && readyMs_) {
        ++quietSamples_;  // still ready: smaller changes neither revoke nor move the anchor off the selected frame
    } else if (!anchor_ || !consecutive || changed || !quiet(*o.anchor, policy_)) {
        anchor_ = current;
        quietSinceMs_ = elapsedMs;
        quietSamples_ = 1;
        ++generation_;
        if (!updateMs_) updateMs_ = elapsedMs;
    } else {
        ++quietSamples_;
    }
    const bool stable = eligible && quietSinceMs_ && elapsedMs - *quietSinceMs_ >= policy_.confirmMs && quietSamples_ >= policy_.minimumConfirmObservations;
    o.newlyReady = stable && !readyMs_ && elapsedMs <= policy_.maximumWaitMs;
    if (o.newlyReady) {
        readyMs_ = elapsedMs;
        anchor_ = current;  // later invalidation compares against the selected frame, not the start of its confirm window
    }
    o.expired = !readyMs_ && elapsedMs >= policy_.maximumWaitMs;
    previous_ = std::move(current);
    previousMs_ = elapsedMs;
    o.quietSinceMs = quietSinceMs_;
    o.quietSamples = quietSamples_;
    o.ready = readyMs_.has_value();
    o.generation = generation_;
    o.firstUpdateMs = updateMs_;
    o.readyMs = readyMs_;
    o.analysisMs = ms(started, std::chrono::steady_clock::now());
    return o;
}

bool Policy::set(const std::string& key, const std::string& value) {
    char* end = nullptr;
    const double v = std::strtod(value.c_str(), &end);
    if (key == "post_acceptance") {
        if (value == "strict") postAcceptance = PostAcceptance::Strict;
        else if (value == "layout_change") postAcceptance = PostAcceptance::LayoutChange;
        else return false;
        return true;
    }
    if (key == "protocol_update_evidence") {
        if (value != "0" && value != "1") return false;
        protocolUpdateEvidence = value == "1";
        return true;
    }
    if (key == "quiet_rule") {
        if (value == "worst_tile") quietRule = QuietRule::WorstTile;
        else if (value == "stable_regions") quietRule = QuietRule::StableRegions;
        else return false;
        return true;
    }
    if (end == value.c_str() || *end) return false;
    const std::map<std::string, double*> doubles{
        {"quiet_ms", &confirmMs}, {"maximum_sample_gap_ms", &maximumSampleGapMs}, {"maximum_wait_ms", &maximumWaitMs},
        {"pixel_threshold", &aliasPixelThreshold}, {"update_mae", &updateMeanDifference}, {"update_fraction", &updateChangedFraction},
        {"blur_sigma", &blurSigma}, {"gradient_threshold", &gradientThreshold}, {"gradient_dominance", &gradientDominance},
        {"minimum_line_fraction", &minimumLineFraction}, {"edge_disagreement", &edgeDisagreement},
        {"tile_disagreement", &cellDisagreement}, {"minimum_stable_regions", &minimumStableCells},
        {"revoke_edge_disagreement", &revokeEdgeDisagreement}};
    const std::map<std::string, int*> ints{
        {"minimum_quiet_samples", &minimumConfirmObservations}, {"minimum_line_pixels", &minimumLinePixels}, {"border_pixels", &borderPixels},
        {"minimum_edge_pixels", &minimumEdgePixels}, {"edge_tolerance_pixels", &edgeTolerancePixels}, {"grid_rows", &gridRows},
        {"grid_columns", &gridColumns}, {"minimum_cell_support", &minimumCellSupport}};
    if (const auto d = doubles.find(key); d != doubles.end()) { *d->second = v; return true; }
    if (const auto i = ints.find(key); i != ints.end()) {
        if (v != std::floor(v)) return false;
        *i->second = static_cast<int>(v);
        return true;
    }
    return false;
}

std::string Policy::describe() const {
    std::ostringstream o;
    o.precision(17);
    o << "{\"quiet_rule\": \"" << (quietRule == QuietRule::WorstTile ? "worst_tile" : "stable_regions") << "\", \"quiet_ms\": " << confirmMs
      << ", \"minimum_quiet_samples\": " << minimumConfirmObservations << ", \"maximum_sample_gap_ms\": " << maximumSampleGapMs
      << ", \"maximum_wait_ms\": " << maximumWaitMs << ", \"pixel_threshold\": " << aliasPixelThreshold << ", \"update_mae\": " << updateMeanDifference
      << ", \"update_fraction\": " << updateChangedFraction << ", \"blur_sigma\": " << blurSigma << ", \"gradient_threshold\": " << gradientThreshold
      << ", \"gradient_dominance\": " << gradientDominance << ", \"minimum_line_fraction\": " << minimumLineFraction
      << ", \"minimum_line_pixels\": " << minimumLinePixels << ", \"border_pixels\": " << borderPixels << ", \"minimum_edge_pixels\": " << minimumEdgePixels
      << ", \"edge_tolerance_pixels\": " << edgeTolerancePixels << ", \"grid_rows\": " << gridRows << ", \"grid_columns\": " << gridColumns
      << ", \"minimum_cell_support\": " << minimumCellSupport << ", \"edge_disagreement\": " << edgeDisagreement
      << ", \"tile_disagreement\": " << cellDisagreement << ", \"minimum_stable_regions\": " << minimumStableCells
      << ", \"protocol_update_evidence\": " << protocolUpdateEvidence << ", \"post_acceptance\": \""
      << (postAcceptance == PostAcceptance::Strict ? "strict" : "layout_change") << "\", \"revoke_edge_disagreement\": " << revokeEdgeDisagreement << "}";
    return o.str();
}

} // namespace readiness
