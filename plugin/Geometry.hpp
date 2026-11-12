// Presenter geometry (M1 Presenter seam): where each leaf of a prepared mesh is drawn at one instant. Pure arithmetic, no GL, no Hyprland --
// checked offline against the acceptance gate's independent model (plugin/test/geometry_check.py).
// Model (experiments/acceptance_gate.py, class Leaf): the old side moves its box from old_box to midpoint_box over t in [0, .5] and is
// frozen there afterwards; the new side leaves midpoint_box at .5 and reaches new_box at 1, frozen at the midpoint before. A mesh leaf's
// vertices are g(h) = lerp(source, midpoint, h), h = 2t (old) or 2(1 - t) (new), drawn at box.xy + g * (size - 1) / extent(h) + .5;
// a plain leaf scales its patch to the box. Everything is divided by the canvas, lerp(old size, new size, t), which is the animated box.
// Differences from the gate's model, all at sub-pixel scale: no rounding to analysis px (the box is in device px; shared leaf edges are
// the same expression on both sides, so they still meet), and the outer vertex ring sits on the box edge in both position and texture
// coordinate, so t = 0 (old) and t = 1 (new) are exactly the identity.
#pragma once

#include "../prep/src/types.hpp"

#include <cstdint>
#include <vector>

namespace Geometry {
    struct SSize {
        double w = 0, h = 0; // analysis px
    };
    struct SVertex {
        float x, y; // position in the animated content box, 0..1, y down
        float u, v; // position in the side's own content, 0..1, y down
    };
    struct SBatch {
        std::vector<SVertex>  vertices;
        std::vector<uint32_t> indices; // triangles
    };

    enum ESide { OLD = 0, NEW = 1 };

    // Appends one side's triangles to `out`. t = geometry progress 0..1; weight = D3-5 mesh weight: positions are
    // lerp(plain scale, mesh, weight), so 0 draws the degenerate mesh (D1-4) whatever the leaves say.
    void build(const std::vector<prep::Leaf>& leaves, SSize oldCanvas, SSize newCanvas, ESide side, double t, double weight, SBatch& out);
}
