#include "Geometry.hpp"

#include <algorithm>
#include <array>

namespace {
    double lerp(double a, double b, double f) {
        return a + (b - a) * f;
    }
}

void Geometry::build(const std::vector<prep::Leaf>& leaves, SSize oldCanvas, SSize newCanvas, ESide side, double t, double weight, SBatch& out) {
    // Each side lives on its own half of t and is frozen at the midpoint on the other half.
    const double tc     = side == OLD ? std::min(t, .5) : std::max(t, .5);
    const double phase  = side == OLD ? 2 * tc : 2 * tc - 1; // box: endpoint -> midpoint (old), midpoint -> endpoint (new)
    const double h      = side == OLD ? 2 * tc : 2 * (1 - tc); // mesh: 0 at the side's own endpoint, 1 at the midpoint
    const SSize  own    = side == OLD ? oldCanvas : newCanvas;
    const SSize  canvas = {lerp(oldCanvas.w, newCanvas.w, tc), lerp(oldCanvas.h, newCanvas.h, tc)};
    const std::array<double, 2> ownSize{own.w, own.h}, canvasSize{canvas.w, canvas.h}, oldSize{oldCanvas.w, oldCanvas.h}, newSize{newCanvas.w, newCanvas.h};

    for (const auto& leaf : leaves) {
        const auto&           source = leaf.box(side);
        std::array<double, 4> box;
        for (size_t k = 0; k < 4; ++k)
            box[k] = side == OLD ? lerp(leaf.oldBox[k], leaf.midpointBox[k], phase) : lerp(leaf.midpointBox[k], leaf.newBox[k], phase);
        // An edge on the canvas edge at both endpoints stays on it: midpoint_box need not be the exact mean of the two canvases.
        for (size_t axis = 0; axis < 2; ++axis) {
            if (leaf.oldBox[axis] == 0 && leaf.newBox[axis] == 0)
                box[axis] = 0;
            if (leaf.oldBox[axis + 2] == oldSize[axis] && leaf.newBox[axis + 2] == newSize[axis])
                box[axis + 2] = canvasSize[axis];
        }
        const std::array<double, 2> size{box[2] - box[0], box[3] - box[1]};
        const auto                  base = static_cast<uint32_t>(out.vertices.size());
        // (sx, sy) in the side's own image, (px, py) on the canvas at t, both analysis px.
        const auto emit = [&](double sx, double sy, double px, double py) {
            const double u = sx / ownSize[0], v = sy / ownSize[1];
            out.vertices.push_back({static_cast<float>(lerp(u, px / canvasSize[0], weight)), static_cast<float>(lerp(v, py / canvasSize[1], weight)),
                                    static_cast<float>(u), static_cast<float>(v)});
        };

        const prep::Grid* grid = nullptr;
        const prep::Grid* middle = nullptr;
        if (leaf.meshes) {
            grid   = &leaf.meshes->source(side);
            middle = &leaf.meshes->midpoint(side);
        }
        const bool meshed = grid && grid->rows >= 2 && grid->cols >= 2 && middle->rows == grid->rows && middle->cols == grid->cols;
        if (!meshed) {
            emit(source[0], source[1], box[0], box[1]);
            emit(source[2], source[1], box[2], box[1]);
            emit(source[0], source[3], box[0], box[3]);
            emit(source[2], source[3], box[2], box[3]);
            for (uint32_t i : {0u, 1u, 3u, 0u, 3u, 2u})
                out.indices.push_back(base + i);
            continue;
        }

        const int                   rows = grid->rows, cols = grid->cols;
        const std::array<double, 2> extent{lerp(grid->x(0, cols - 1), middle->x(0, cols - 1), h), lerp(grid->y(rows - 1, 0), middle->y(rows - 1, 0), h)};
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                // Grid coordinates are pixel centres within the patch; the outer ring is pulled out to the patch edge (half a pixel).
                double sx = source[0] + grid->x(r, c) + .5, sy = source[1] + grid->y(r, c) + .5;
                double px = box[0] + lerp(grid->x(r, c), middle->x(r, c), h) * (size[0] - 1) / extent[0] + .5;
                double py = box[1] + lerp(grid->y(r, c), middle->y(r, c), h) * (size[1] - 1) / extent[1] + .5;
                if (c == 0)
                    sx = source[0], px = box[0];
                if (c == cols - 1)
                    sx = source[2], px = box[2];
                if (r == 0)
                    sy = source[1], py = box[1];
                if (r == rows - 1)
                    sy = source[3], py = box[3];
                emit(sx, sy, px, py);
            }
        for (int r = 0; r + 1 < rows; ++r)
            for (int c = 0; c + 1 < cols; ++c) {
                const uint32_t a = base + static_cast<uint32_t>(r * cols + c), b = a + 1, d = a + static_cast<uint32_t>(cols) + 1, e = a + static_cast<uint32_t>(cols);
                for (uint32_t i : {a, b, d, a, d, e})
                    out.indices.push_back(i);
            }
    }
}
