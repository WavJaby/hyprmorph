#pragma once
// Leaf = one partition region as the Presenter draws it; toJson/leafFromJson define the worker payload's leaf shape.
// Standard library only: the compositor plugin compiles this file and types.cpp to read the worker's payload.
#include "json.hpp"

#include <array>
#include <cstdint>
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

// Clip container anchor per axis (x, y): the end of the leaf box its content stays attached to.
enum class Anchor : uint8_t { Start, End };

struct Leaf {
    std::string name, kind;
    Box oldBox{}, newBox{};
    BoxF midpointBox{};
    // Set = clip container (M4 D4-4): old and new agree pixel-for-pixel over their overlap placed at these anchors, and one box contains
    // the other's size. Drawn 1:1 from the larger image, anchored, cropped to the box; never meshed.
    std::optional<std::array<Anchor, 2>> clip;
    std::optional<Meshes> meshes;
    // Rigid-strip result; windows = fixed_windows_vertex_indices [c0, r0, c1, r1] (vertex indices), others are evidence.
    std::optional<json::Value> rigid;
    std::vector<std::array<int, 4>> windows;
    const Box& box(int side) const { return side ? newBox : oldBox; }
    // Clip containers only: the side whose box is the larger (detection guarantees it contains the other's size on both axes).
    int clipSource() const { return newBox[2] - newBox[0] + newBox[3] - newBox[1] >= oldBox[2] - oldBox[0] + oldBox[3] - oldBox[1]; }
};

json::Value toJson(const Grid& grid);
Grid gridFromJson(const json::Value& value);
json::Value toJson(const Leaf& leaf);
Leaf leafFromJson(const json::Value& value);
json::Value toJson(const std::vector<Leaf>& leaves);
std::vector<Leaf> leavesFromJson(const json::Value& value);

} // namespace prep
