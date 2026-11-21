// Presenter geometry: where each leaf of a prepared mesh is drawn at one instant. Pure arithmetic, no GL, no Hyprland.
// Model: the old side moves its box from old_box to midpoint_box over t in [0, .5] and is
// frozen there afterwards; the new side leaves midpoint_box at .5 and reaches new_box at 1, frozen at the midpoint before. A mesh leaf's
// vertices are g(h) = lerp(source, midpoint, h), h = 2t (old) or 2(1 - t) (new), drawn at box.xy + g * (size - 1) / extent(h) + .5;
// a plain leaf scales its patch to the box. Everything is divided by the canvas, lerp(old size, new size, t), which is the animated box.
// Departures from that model as stated in analysis px, all at sub-pixel scale: no rounding to analysis px (the box is in device px; shared leaf edges are
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
    // MORPH: the leaves that deform, drawn from `side`'s image in that side's crossfade layer. CLIP: the clip containers whose larger
    // image is `side` (prep::Leaf::clipSource), drawn at full opacity in a layer of their own: no crossfade inside them.
    enum EPart { MORPH = 0, CLIP = 1 };

    // Appends one side's triangles of one part to `out`. t = geometry progress 0..1; weight = mesh weight: positions are
    // lerp(plain scale, mesh, weight), so 0 draws the degenerate mesh whatever the leaves say.
    // A clip leaf: box on the shared path, content = its source image 1:1 from the anchored end, cropped to the box (texture window
    // [s0, s0 + L] or [s1 - L, s1] per axis, L = box length at t in canvas px).
    void build(const std::vector<prep::Leaf>& leaves, SSize oldCanvas, SSize newCanvas, ESide side, EPart part, double t, double weight, SBatch& out);
}
