#pragma once
// Leaf = one partition region as the Presenter draws it (same JSON shape as experiments/acceptance_pipeline.py).
// Standard library only: the compositor plugin compiles this file and types.cpp to read the worker's payload.
#include "json.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace prep {

using Box = std::array<int, 4>;        // x0, y0, x1, y1 (exclusive), analysis px
using BoxF = std::array<double, 4>;

// rows x cols vertices, (x, y) each; patch pixel-centre coordinates.
struct Grid {
    int rows = 0, cols = 0;
    std::vector<double> xy;
    Grid() = default;
    Grid(int r, int c) : rows(r), cols(c), xy(size_t(r) * size_t(c) * 2) {}
    double& x(int r, int c) { return xy[(size_t(r) * size_t(cols) + size_t(c)) * 2]; }
    double& y(int r, int c) { return xy[(size_t(r) * size_t(cols) + size_t(c)) * 2 + 1]; }
    double x(int r, int c) const { return xy[(size_t(r) * size_t(cols) + size_t(c)) * 2]; }
    double y(int r, int c) const { return xy[(size_t(r) * size_t(cols) + size_t(c)) * 2 + 1]; }
    double& at(int r, int c, int d) { return xy[(size_t(r) * size_t(cols) + size_t(c)) * 2 + size_t(d)]; }
    double at(int r, int c, int d) const { return xy[(size_t(r) * size_t(cols) + size_t(c)) * 2 + size_t(d)]; }
};

struct Meshes {
    Grid oldSource, oldMidpoint, newSource, newMidpoint;
    Grid& source(int side) { return side ? newSource : oldSource; }
    Grid& midpoint(int side) { return side ? newMidpoint : oldMidpoint; }
    const Grid& source(int side) const { return side ? newSource : oldSource; }
    const Grid& midpoint(int side) const { return side ? newMidpoint : oldMidpoint; }
};

struct Leaf {
    std::string name, kind;
    Box oldBox{}, newBox{};
    BoxF midpointBox{};
    std::optional<Meshes> meshes;
    // Rigid-strip result; windows = fixed_windows_vertex_indices [c0, r0, c1, r1] (vertex indices), others are evidence.
    std::optional<json::Value> rigid;
    std::vector<std::array<int, 4>> windows;
    const Box& box(int side) const { return side ? newBox : oldBox; }
};

json::Value toJson(const Grid& grid);
Grid gridFromJson(const json::Value& value);
json::Value toJson(const Leaf& leaf);
Leaf leafFromJson(const json::Value& value);
json::Value toJson(const std::vector<Leaf>& leaves);
std::vector<Leaf> leavesFromJson(const json::Value& value);

} // namespace prep
