#pragma once
// Fixed-iteration regularised conjugate gradient on GLES compute (port of paired_point_constraints.solve_gpu):
// minimise |A x - b|^2 + .08 |x|^2 with 96 CG iterations from x = 0, float32, no CPU fallback.
// The reference builds A dense (<= 6 non-zeros per row); here A is CSR + its transpose, one workgroup per system, all
// iterations inside one dispatch, every system of a call batched into that dispatch.
#include "gles.hpp"

#include <memory>
#include <vector>

namespace prep {

struct SparseSystem {
    int rows = 0, cols = 0;
    std::vector<int> rowPtr{0}, colIdx;  // CSR of A
    std::vector<float> val, rhs;         // rhs = b (length rows)
    // Builds one row: duplicates are summed in double, then rounded to float32 (the reference forms the dense matrix
    // in float64 and casts it with .astype(np.float32)).
    void add(int col, double value);
    void endRow(double b);
    static SparseSystem dense(const float* a, int rows, int cols, const float* b);

private:
    std::vector<std::pair<int, double>> pending;
};

class CgSolver {
public:
    static constexpr int ITERATIONS = 96;
    static constexpr float REGULARISER = .08f;
    explicit CgSolver(gles::Context& context);
    ~CgSolver();
    std::vector<std::vector<float>> solve(const std::vector<SparseSystem>& systems);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace prep
