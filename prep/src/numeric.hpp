#pragma once
// NumPy-equivalent reductions where the reference's discrete decisions depend on them.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace prep::numeric {

// np.median: middle element, or the mean of the two middle elements computed in T (float32 for float32 input).
template <typename T>
T median(std::vector<T> values) {
    size_t n = values.size();
    std::sort(values.begin(), values.end());
    if (n % 2) return values[n / 2];
    return T(values[n / 2 - 1] + values[n / 2]) / T(2);
}

// np.percentile(..., method='linear') with NumPy's _lerp (b - diff*(1-t) for t >= .5).
inline double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    double index = q / 100. * double(values.size() - 1);
    size_t lo = size_t(std::floor(index)), hi = std::min(lo + 1, values.size() - 1);
    double t = index - double(lo), a = values[lo], b = values[hi], diff = b - a;
    return t >= .5 ? b - diff * (1 - t) : a + diff * t;
}

// NumPy pairwise summation (blocks of 128, 8 accumulators) over a contiguous range.
template <typename T>
T pairwise(const T* a, size_t n) {
    if (n < 8) {
        T res = 0;
        for (size_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        T r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + size_t(j)];
        T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise(a, n2) + pairwise(a + n2, n - n2);
}

// np.sum of a 1-D contiguous array: the reduction starts from the identity 0 and adds the pairwise sum of all items.
template <typename T>
T sum(const std::vector<T>& values) {
    return pairwise(values.data(), values.size());
}

template <typename T>
T mean(const std::vector<T>& values) {
    return sum(values) / T(values.size());
}

inline bool isclose(double a, double b, double rtol = 1e-5, double atol = 1e-8) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

} // namespace prep::numeric
