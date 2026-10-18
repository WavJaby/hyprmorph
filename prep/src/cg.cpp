#include "cg.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace prep {
namespace {

constexpr const char* SOURCE = R"GLSL(#version 320 es
precision highp float;
precision highp int;
layout(local_size_x=256) in;
struct System { int rows, cols, valOff, rowOff, colOff, vecOff, yOff, bOff; };
layout(std430, binding=0) readonly buffer Systems { System systems[]; };
layout(std430, binding=1) readonly buffer Index { int index[]; };   // rowPtr | colIdx | colPtr | rowIdx per system
layout(std430, binding=2) readonly buffer Values { float values[]; };  // val | valT per system
layout(std430, binding=3) coherent buffer Work { float work[]; };      // b, y (rows) | x, r, p, ap (cols)
shared float partial[256];
float reduce(float value) {
    uint lane = gl_LocalInvocationID.x;
    partial[lane] = value;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        barrier();
    }
    float total = partial[0];
    barrier();
    return total;
}
void main() {
    System s = systems[gl_WorkGroupID.x];
    int lane = int(gl_LocalInvocationID.x);
    int m = s.rows, n = s.cols, nnz = index[s.rowOff + m];
    int rowPtr = s.rowOff, colIdx = s.rowOff + m + 1, colPtr = colIdx + nnz, rowIdx = colPtr + n + 1;
    int val = s.valOff, valT = s.valOff + nnz;
    int X = s.vecOff, R = X + n, P = R + n, AP = P + n, Y = s.yOff, B = s.bOff;
    // rhs = A^T b; x = 0, r = p = rhs
    float local = 0.0;
    for (int j = lane; j < n; j += 256) {
        float acc = 0.0;
        for (int k = index[colPtr + j]; k < index[colPtr + j + 1]; ++k) acc += values[valT + k] * work[B + index[rowIdx + k]];
        work[X + j] = 0.0; work[R + j] = acc; work[P + j] = acc;
        local += acc * acc;
    }
    float rr = reduce(local);
    memoryBarrierBuffer();
    barrier();
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        for (int i = lane; i < m; i += 256) {
            float acc = 0.0;
            for (int k = index[rowPtr + i]; k < index[rowPtr + i + 1]; ++k) acc += values[val + k] * work[P + index[colIdx + k]];
            work[Y + i] = acc;
        }
        memoryBarrierBuffer();
        barrier();
        local = 0.0;
        for (int j = lane; j < n; j += 256) {
            float acc = 0.0;
            for (int k = index[colPtr + j]; k < index[colPtr + j + 1]; ++k) acc += values[valT + k] * work[Y + index[rowIdx + k]];
            float ap = acc + REGULARISER * work[P + j];
            work[AP + j] = ap;
            local += work[P + j] * ap;
        }
        float pap = reduce(local);
        float alpha = rr / max(pap, 1e-20);
        local = 0.0;
        for (int j = lane; j < n; j += 256) {
            work[X + j] += alpha * work[P + j];
            float r = work[R + j] - alpha * work[AP + j];
            work[R + j] = r;
            local += r * r;
        }
        float fresh = reduce(local);
        float beta = fresh / max(rr, 1e-20);
        for (int j = lane; j < n; j += 256) work[P + j] = work[R + j] + beta * work[P + j];
        rr = fresh;
        memoryBarrierBuffer();
        barrier();
    }
}
)GLSL";

struct System {
    int32_t rows, cols, valOff, rowOff, colOff, vecOff, yOff, bOff;
};

} // namespace

void SparseSystem::add(int col, double value) { pending.emplace_back(col, value); }

void SparseSystem::endRow(double b) {
    std::stable_sort(pending.begin(), pending.end(), [](auto& x, auto& y) { return x.first < y.first; });
    for (size_t i = 0; i < pending.size();) {
        double total = 0;
        size_t j = i;
        for (; j < pending.size() && pending[j].first == pending[i].first; ++j) total += pending[j].second;
        float value = float(total);
        if (value != 0) {
            colIdx.push_back(pending[i].first);
            val.push_back(value);
        }
        i = j;
    }
    pending.clear();
    rowPtr.push_back(int(colIdx.size()));
    rhs.push_back(float(b));
    ++rows;
}

SparseSystem SparseSystem::dense(const float* a, int rows, int cols, const float* b) {
    SparseSystem s;
    s.cols = cols;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            if (a[size_t(i) * size_t(cols) + size_t(j)] != 0) s.add(j, a[size_t(i) * size_t(cols) + size_t(j)]);
        s.endRow(b[i]);
    }
    return s;
}

struct CgSolver::Impl {
    GLuint program = 0;
    gles::Buffer systems, index, values, work;
};

CgSolver::CgSolver(gles::Context&) : impl(std::make_unique<Impl>()) {
    std::string source = SOURCE;
    // The literal must equal float32(.08): print it with full precision.
    char literal[64];
    std::snprintf(literal, sizeof literal, "#define REGULARISER %.9g\n", double(REGULARISER));
    std::string defines = "#define ITERATIONS " + std::to_string(ITERATIONS) + "\n" + literal;
    source.insert(source.find('\n') + 1, defines);
    impl->program = gles::program(source.c_str());
}

CgSolver::~CgSolver() { glDeleteProgram(impl->program); }

std::vector<std::vector<float>> CgSolver::solve(const std::vector<SparseSystem>& input) {
    std::vector<std::vector<float>> out(input.size());
    std::vector<System> table;
    std::vector<int32_t> index;
    std::vector<float> values, work;
    std::vector<size_t> owner;
    for (size_t s = 0; s < input.size(); ++s) {
        const SparseSystem& a = input[s];
        if (a.cols == 0) continue;
        int m = a.rows, n = a.cols, nnz = int(a.colIdx.size());
        // Transpose (CSC) for A^T y; rows inside a column stay ascending.
        std::vector<int> colPtr(size_t(n) + 1, 0), rowIdx(static_cast<size_t>(nnz));
        std::vector<float> valT(static_cast<size_t>(nnz));
        for (int k = 0; k < nnz; ++k) ++colPtr[size_t(a.colIdx[size_t(k)]) + 1];
        for (int j = 0; j < n; ++j) colPtr[size_t(j) + 1] += colPtr[size_t(j)];
        std::vector<int> fill(colPtr.begin(), colPtr.end() - 1);
        for (int i = 0; i < m; ++i)
            for (int k = a.rowPtr[size_t(i)]; k < a.rowPtr[size_t(i) + 1]; ++k) {
                int slot = fill[size_t(a.colIdx[size_t(k)])]++;
                rowIdx[size_t(slot)] = i;
                valT[size_t(slot)] = a.val[size_t(k)];
            }
        System sys{m, n, int32_t(values.size()), int32_t(index.size()), 0, 0, 0, 0};
        index.insert(index.end(), a.rowPtr.begin(), a.rowPtr.end());
        index.insert(index.end(), a.colIdx.begin(), a.colIdx.end());
        index.insert(index.end(), colPtr.begin(), colPtr.end());
        index.insert(index.end(), rowIdx.begin(), rowIdx.end());
        values.insert(values.end(), a.val.begin(), a.val.end());
        values.insert(values.end(), valT.begin(), valT.end());
        sys.bOff = int32_t(work.size());
        work.insert(work.end(), a.rhs.begin(), a.rhs.end());
        sys.yOff = int32_t(work.size());
        work.resize(work.size() + size_t(m));
        sys.vecOff = int32_t(work.size());
        work.resize(work.size() + 4 * size_t(n));
        table.push_back(sys);
        owner.push_back(s);
    }
    if (table.empty()) return out;
    Impl& g = *impl;
    g.systems.upload(table.data(), table.size() * sizeof(System));
    g.index.upload(index.data(), index.size() * 4);
    g.values.upload(values.data(), values.size() * 4);
    g.work.upload(work.data(), work.size() * 4);
    g.systems.bind(0);
    g.index.bind(1);
    g.values.bind(2);
    g.work.bind(3);
    glUseProgram(g.program);
    glDispatchCompute(GLuint(table.size()), 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    g.work.download(work.data(), work.size() * 4);
    gles::check("cg solve");
    for (size_t k = 0; k < table.size(); ++k) {
        const System& s = table[k];
        out[owner[k]].assign(work.begin() + s.vecOff, work.begin() + s.vecOff + s.cols);
    }
    return out;
}

} // namespace prep
