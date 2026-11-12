// Presenter, GL half (M1 Presenter seam): draws the triangles Geometry built into a framebuffer, sampling the old snapshot and the live new
// content. Layers are summed with their opacities (premultiplied crossfade: old * (1 - f) + new * f), so a D3-1 fade is exact whatever
// the content's own alpha. Rows follow the snapshots' convention: framebuffer row 0 is the top of the content.
// Needs the compositor's EGL context current; every GL state it touches is put back.
#pragma once

#include "Geometry.hpp"

#include <span>

namespace Presenter {
    struct SLayer {
        unsigned                texture = 0;
        float                   uExtent = 1, vExtent = 1; // the content's share of the texture: batch u, v = 1 lands here
        const Geometry::SBatch* batch   = nullptr;
        float                   opacity = 1;
    };

    // Clears (0, 0)-(width, height) of `framebuffer` to transparent and draws the layers into it, batch position 0..1 spanning that rect.
    // false -> GL setup failed, reason in failure(); nothing usable was drawn.
    bool        compose(unsigned framebuffer, int width, int height, std::span<const SLayer> layers);
    void        destroy();
    const char* failure(); // empty while working
}
