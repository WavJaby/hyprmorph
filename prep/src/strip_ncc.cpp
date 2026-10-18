#include "strip_ncc.hpp"

#include <algorithm>
#include <cstring>

namespace prep {
namespace {

constexpr const char* COMMON = R"GLSL(#version 320 es
precision highp float;
precision highp int;
struct Job { int Wa, Ha, Wb, Hb, height, oldY, y0, rows; uint aOff, bOff, cOff, colOff; };
struct Patch { int job, x, width, n; uint sp, sp2; int first, count; };
layout(std430, binding=0) readonly buffer Jobs { Job jobs[]; };
layout(std430, binding=1) readonly buffer Pixels { uint pixels[]; };
layout(std430, binding=2) readonly buffer Cols { uint cols[]; };
layout(std430, binding=4) readonly buffer Patches { Patch patches[]; };
)GLSL";

// Column-correlation table: per job and allowed row r, C[xa][xb] = sum_{dy<height, c} a[oldY+dy][xa][c] * b[y0+r+dy][xb][c].
constexpr const char* TABLE = R"GLSL(
layout(std430, binding=3) writeonly buffer Table { uint table[]; };
layout(std430, binding=6) readonly buffer Rows { ivec2 rowJob[]; };
layout(local_size_x=16, local_size_y=16) in;
shared uint ta[16][17];
shared uint tb[16][17];
void main() {
    ivec2 rj = rowJob[gl_WorkGroupID.z];
    Job j = jobs[rj.x];
    int r = rj.y;
    int tx = int(gl_LocalInvocationID.x), ty = int(gl_LocalInvocationID.y);
    int xb0 = int(gl_WorkGroupID.x) * 16, xa0 = int(gl_WorkGroupID.y) * 16;
    if (xb0 >= j.Wb || xa0 >= j.Wa) return;
    int K = j.height * 3, yy = j.y0 + r;
    uint acc = 0u;
    for (int k0 = 0; k0 < K; k0 += 16) {
        int k = k0 + tx, xa = xa0 + ty;
        ta[ty][tx] = (k < K && xa < j.Wa) ? pixels[j.aOff + uint(((j.oldY + k / 3) * j.Wa + xa) * 3 + k % 3)] : 0u;
        k = k0 + ty;
        int xb = xb0 + tx;
        tb[ty][tx] = (k < K && xb < j.Wb) ? pixels[j.bOff + uint(((yy + k / 3) * j.Wb + xb) * 3 + k % 3)] : 0u;
        barrier();
        for (int kk = 0; kk < 16; ++kk) acc += ta[ty][kk] * tb[kk][tx];
        barrier();
    }
    int xa = xa0 + ty, xb = xb0 + tx;
    if (xa < j.Wa && xb < j.Wb) table[j.cOff + uint(r * j.Wa + xa) * uint(j.Wb) + uint(xb)] = acc;
}
)GLSL";

// Exact integer NCC terms in emulated 64-bit, ratio in float.
constexpr const char* SCORE = R"GLSL(
layout(std430, binding=3) readonly buffer Table { uint table[]; };
layout(std430, binding=5) buffer Results { uvec4 results[]; };
uvec2 mul64(uint a, uint b) { uint hi, lo; umulExtended(a, b, hi, lo); return uvec2(lo, hi); }
uvec2 sub64(uvec2 a, uvec2 b) { uint borrow; uint lo = usubBorrow(a.x, b.x, borrow); return uvec2(lo, a.y - b.y - borrow); }
float toFloat(uvec2 v) {
    bool negative = (v.y & 0x80000000u) != 0u;
    if (negative) v = sub64(uvec2(0u), v);
    float f = float(v.y) * 4294967296.0 + float(v.x);
    return negative ? -f : f;
}
float ncc(Patch p, Job j, int r, int xx) {
    uint dot = 0u, sc = 0u, sc2 = 0u;
    uint row = j.cOff + uint(r * j.Wa) * uint(j.Wb);
    uint col = j.colOff + uint(r * j.Wb + xx);
    uint plane = uint(j.rows * j.Wb);
    for (int dx = 0; dx < p.width; ++dx) {
        dot += table[row + uint(p.x + dx) * uint(j.Wb) + uint(xx + dx)];
        sc += cols[col + uint(dx)];
        sc2 += cols[col + plane + uint(dx)];
    }
    uint n = uint(p.n);
    uvec2 dc = sub64(mul64(n, sc2), mul64(sc, sc));
    if (dc == uvec2(0u)) return 0.0;
    float num = toFloat(sub64(mul64(n, dot), mul64(p.sp, sc)));
    float dp = toFloat(sub64(mul64(n, p.sp2), mul64(p.sp, p.sp)));
    return num / sqrt(dp * toFloat(dc));
}
)GLSL";

// Per patch: peak over allowed positions (first index on ties, row-major like argmax), then the best score outside the
// peak's +-max(4, width/2) column band (masked positions count as -1, as in the reference).
constexpr const char* PEAK = R"GLSL(
layout(local_size_x=256) in;
shared float sv[256];
shared int si[256];
void main() {
    uint pid = gl_WorkGroupID.x;
    Patch p = patches[pid];
    Job j = jobs[p.job];
    int lane = int(gl_LocalInvocationID.x);
    int ncols = j.Wb - p.width + 1;
    int xlo = 2, span = j.Wb - p.width - 2 - xlo + 1;
    int total = max(span, 0) * j.rows;
    float best = -1.0;
    int index = 0x7fffffff;
    for (int k = lane; k < total; k += 256) {
        int r = k / span, xx = xlo + k % span;
        float s = ncc(p, j, r, xx);
        if (s > best || index == 0x7fffffff) { best = s; index = (j.y0 + r) * ncols + xx; }
    }
    sv[lane] = best; si[lane] = index;
    barrier();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            float v = sv[lane + stride]; int i = si[lane + stride];
            if (i != 0x7fffffff && (si[lane] == 0x7fffffff || v > sv[lane] || (v == sv[lane] && i < si[lane]))) { sv[lane] = v; si[lane] = i; }
        }
        barrier();
    }
    float peak = si[0] == 0x7fffffff ? -1.0 : sv[0];
    int peakIndex = si[0] == 0x7fffffff ? 0 : si[0];
    barrier();
    int bestx = peakIndex % ncols, exclude = max(4, p.width / 2);
    float second = -1.0;
    for (int k = lane; k < total; k += 256) {
        int r = k / span, xx = xlo + k % span;
        if (abs(xx - bestx) > exclude) second = max(second, ncc(p, j, r, xx));
    }
    sv[lane] = second;
    barrier();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) sv[lane] = max(sv[lane], sv[lane + stride]);
        barrier();
    }
    if (lane == 0) results[pid] = uvec4(floatBitsToUint(peak), floatBitsToUint(sv[0]), uint(peakIndex), 0u);
}
)GLSL";

// Per patch with a usable peak: which patch of the same width scores best at that position (argmax over patches).
constexpr const char* REVERSE = R"GLSL(
layout(local_size_x=64) in;
shared float sv[64];
shared int si[64];
void main() {
    uint pid = gl_WorkGroupID.x;
    Patch p = patches[pid];
    Job j = jobs[p.job];
    int lane = int(gl_LocalInvocationID.x);
    uvec4 own = results[pid];
    if (uintBitsToFloat(own.x) < 0.9) return;
    int ncols = j.Wb - p.width + 1;
    int index = int(own.z), r = index / ncols - j.y0, xx = index % ncols;
    float best = -2.0;
    int arg = 0x7fffffff;
    for (int q = lane; q < p.count; q += 64) {
        float s = ncc(patches[p.first + q], j, r, xx);
        if (s > best) { best = s; arg = q; }
    }
    sv[lane] = best; si[lane] = arg;
    barrier();
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            float v = sv[lane + stride]; int i = si[lane + stride];
            if (v > sv[lane] || (v == sv[lane] && i < si[lane])) { sv[lane] = v; si[lane] = i; }
        }
        barrier();
    }
    if (lane == 0) results[pid].w = uint(si[0]);
}
)GLSL";

struct Job {
    int32_t Wa, Ha, Wb, Hb, height, oldY, y0, rows;
    uint32_t aOff, bOff, cOff, colOff;
};
struct Patch {
    int32_t job, x, width, n;
    uint32_t sp, sp2;
    int32_t first, count;
};
constexpr int WIDTHS[] = {6, 8, 12, 16};

} // namespace

struct StripNcc::Impl {
    GLuint table = 0, peak = 0, reverse = 0;
    gles::Buffer jobs, pixels, cols, cells, patches, results, rows;
};

StripNcc::StripNcc(gles::Context&) : impl(std::make_unique<Impl>()) {
    impl->table = gles::program((std::string(COMMON) + TABLE).c_str());
    impl->peak = gles::program((std::string(COMMON) + SCORE + PEAK).c_str());
    impl->reverse = gles::program((std::string(COMMON) + SCORE + REVERSE).c_str());
}

StripNcc::~StripNcc() {
    glDeleteProgram(impl->table);
    glDeleteProgram(impl->peak);
    glDeleteProgram(impl->reverse);
}

std::vector<StripScores> StripNcc::score(const std::vector<std::pair<cv::Mat, cv::Mat>>& inputs) {
    std::vector<StripScores> out(inputs.size());
    std::vector<Job> jobs;
    std::vector<int> jobOf(inputs.size(), -1);
    std::vector<uint32_t> pixels, cols;
    std::vector<Patch> patches;
    std::vector<int32_t> rowJob;
    std::vector<std::pair<int, int>> owner;  // patch -> (input, index in out.patches)
    size_t cells = 0;
    int maxWa = 0, maxWb = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const cv::Mat& a = inputs[i].first;
        const cv::Mat& b = inputs[i].second;
        CV_Assert(a.type() == CV_8UC3 && b.type() == CV_8UC3 && a.isContinuous() && b.isContinuous());
        StripScores& result = out[i];
        int height = std::min(a.rows, b.rows) - 6;
        result.height = height;
        if (height < 5) continue;
        int oldY = (a.rows - height) / 2;
        result.oldY = oldY;
        int y0 = std::max(1, oldY - 4), y1 = std::min(b.rows - height - 1, oldY + 4);
        int nrows = std::max(0, y1 - y0 + 1);
        // Column sums over the patch rows (old) and over each allowed window (new): exact integers.
        std::vector<uint32_t> colA(size_t(a.cols)), colA2(size_t(a.cols));
        for (int dy = 0; dy < height; ++dy) {
            const uint8_t* row = a.ptr<uint8_t>(oldY + dy);
            for (int x = 0; x < a.cols; ++x)
                for (int c = 0; c < 3; ++c) {
                    uint32_t v = row[x * 3 + c];
                    colA[size_t(x)] += v;
                    colA2[size_t(x)] += v * v;
                }
        }
        int firstPatch = int(patches.size());
        for (int width : WIDTHS) {
            if (std::min(a.cols, b.cols) < width + 8) continue;
            int first = int(patches.size());
            uint32_t n = uint32_t(height * width * 3);
            for (int x = 3; x < a.cols - width - 2; x += 3) {
                uint64_t sp = 0, sp2 = 0;
                for (int dx = 0; dx < width; ++dx) {
                    sp += colA[size_t(x + dx)];
                    sp2 += colA2[size_t(x + dx)];
                }
                // std > 12 <=> n*sum(x^2) - sum(x)^2 > 144 n^2 (exact; the reference computes float32 std).
                if (int64_t(n) * int64_t(sp2) - int64_t(sp) * int64_t(sp) <= int64_t(144) * n * n) continue;
                patches.push_back({0, x, width, int32_t(n), uint32_t(sp), uint32_t(sp2), first, 0});
                owner.emplace_back(int(i), int(result.patches.size()));
                PatchScore score;
                score.x = x;
                score.width = width;
                result.patches.push_back(score);
            }
            for (size_t k = size_t(first); k < patches.size(); ++k) patches[k].count = int32_t(patches.size()) - first;
        }
        if (nrows == 0 || patches.size() == size_t(firstPatch)) {
            patches.resize(size_t(firstPatch));
            owner.resize(size_t(firstPatch));
            continue;  // no allowed position: every peak is -1, no candidate
        }
        int job = int(jobs.size());
        jobOf[i] = job;
        for (size_t k = size_t(firstPatch); k < patches.size(); ++k) patches[k].job = job;
        Job j{a.cols, a.rows, b.cols, b.rows, height, oldY, y0, nrows,
              uint32_t(pixels.size()), uint32_t(pixels.size() + a.total() * 3), uint32_t(cells), uint32_t(cols.size())};
        pixels.insert(pixels.end(), a.data, a.data + a.total() * 3);
        pixels.insert(pixels.end(), b.data, b.data + b.total() * 3);
        std::vector<uint32_t> colB(size_t(nrows) * size_t(b.cols)), colB2(colB.size());
        for (int r = 0; r < nrows; ++r)
            for (int dy = 0; dy < height; ++dy) {
                const uint8_t* row = b.ptr<uint8_t>(y0 + r + dy);
                for (int x = 0; x < b.cols; ++x)
                    for (int c = 0; c < 3; ++c) {
                        uint32_t v = row[x * 3 + c];
                        colB[size_t(r * b.cols + x)] += v;
                        colB2[size_t(r * b.cols + x)] += v * v;
                    }
            }
        cols.insert(cols.end(), colB.begin(), colB.end());
        cols.insert(cols.end(), colB2.begin(), colB2.end());
        for (int r = 0; r < nrows; ++r) {
            rowJob.push_back(job);
            rowJob.push_back(r);
        }
        cells += size_t(nrows) * size_t(a.cols) * size_t(b.cols);
        maxWa = std::max(maxWa, a.cols);
        maxWb = std::max(maxWb, b.cols);
        jobs.push_back(j);
    }
    if (jobs.empty()) return out;
    Impl& g = *impl;
    g.jobs.upload(jobs.data(), jobs.size() * sizeof(Job));
    g.pixels.upload(pixels.data(), pixels.size() * 4);
    g.cols.upload(cols.data(), cols.size() * 4);
    g.cells.reserve(cells * 4);
    g.patches.upload(patches.data(), patches.size() * sizeof(Patch));
    g.results.reserve(patches.size() * 16);
    g.rows.upload(rowJob.data(), rowJob.size() * 4);
    g.jobs.bind(0);
    g.pixels.bind(1);
    g.cols.bind(2);
    g.cells.bind(3);
    g.patches.bind(4);
    g.results.bind(5);
    g.rows.bind(6);
    glUseProgram(g.table);
    glDispatchCompute(GLuint((maxWb + 15) / 16), GLuint((maxWa + 15) / 16), GLuint(rowJob.size() / 2));
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUseProgram(g.peak);
    glDispatchCompute(GLuint(patches.size()), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    glUseProgram(g.reverse);
    glDispatchCompute(GLuint(patches.size()), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    std::vector<uint32_t> results(patches.size() * 4);
    g.results.download(results.data(), results.size() * 4);
    gles::check("strip ncc");
    for (size_t k = 0; k < patches.size(); ++k) {
        auto [input, index] = owner[k];
        PatchScore& s = out[size_t(input)].patches[size_t(index)];
        const Job& j = jobs[size_t(patches[k].job)];
        int ncols = j.Wb - s.width + 1;
        std::memcpy(&s.peak, &results[k * 4], 4);
        std::memcpy(&s.second, &results[k * 4 + 1], 4);
        int position = int(results[k * 4 + 2]);
        s.bestx = position % ncols;
        s.besty = position / ncols;
        s.reverse = s.peak >= 0.9f ? int(results[k * 4 + 3]) : 0;
    }
    return out;
}

} // namespace prep
