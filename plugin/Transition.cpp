#include "Transition.hpp"

#include "Presenter.hpp"

#include <hyprland/src/debug/log/Logger.hpp>
#include <hyprland/src/desktop/view/Window.hpp>
#include <hyprland/src/output/Monitor.hpp>
#include <hyprland/src/protocols/core/Compositor.hpp>
#include <hyprland/src/protocols/XDGShell.hpp>
#include <hyprland/src/render/Renderer.hpp>
#include <hyprland/src/render/gl/GLFramebuffer.hpp>
#include <hyprland/src/render/pass/ClearPassElement.hpp>
#include <hyprland/src/render/pass/TexPassElement.hpp>

#include <algorithm>

using namespace std::chrono;

namespace {
    // Hyprland's "no custom UV" sentinel for m_renderData.primarySurfaceUV*.
    const Vector2D UV_UNSET{-1, -1};

    void setCustomUV(const Vector2D& topLeft, const Vector2D& bottomRight) {
        auto& rd                       = g_pHyprRenderer->m_renderData;
        rd.primarySurfaceUVTopLeft     = topLeft;
        rd.primarySurfaceUVBottomRight = bottomRight;
    }

    double msSince(steady_clock::time_point t, steady_clock::time_point now) {
        return duration<double, std::milli>(now - t).count();
    }
}


CResizeTransition::CResizeTransition(PHLWINDOWREF window) : m_window(window) {}

CResizeTransition::~CResizeTransition() = default;

// Renders the window's committed surface tree at its own size into out.fb, content-only (no decorations, no rounding, no popups).
// Called outside any render pass: from precommit (client dispatch) or render.preChecks (before beginRender), like makeSnapshotFB() at unmap.
bool CResizeTransition::capture(SSnapshot& out) {
    const auto window = m_window.lock();
    if (!window || !window->m_isMapped) {
        m_failure = "window gone";
        return false;
    }
    // F11 scope: X11 has no xdg ack to detect the new buffer; rotated outputs break the monitor-local crop below.
    if (window->m_isX11 || !window->m_xdgSurface) {
        m_failure = "X11 window";
        return false;
    }
    const auto monitor = window->m_monitor.lock();
    if (!monitor || !monitor->m_output || monitor->m_pixelSize.x <= 0 || monitor->m_pixelSize.y <= 0) {
        m_failure = "no monitor";
        return false;
    }
    if (monitor->m_transform != WL_OUTPUT_TRANSFORM_NORMAL) {
        m_failure = "rotated monitor";
        return false;
    }
    const auto surface = window->resource();
    if (!surface || !surface->m_current.texture) {
        m_failure = "no committed buffer";
        return false;
    }

    // Content box = xdg geometry (excludes client-side shadows); maps 1:1 onto the window box when presented.
    CBox geometry = window->m_xdgSurface->m_current.geometry;
    if (geometry.empty())
        geometry = CBox{{}, surface->m_current.size};
    const Vector2D contentPx = (geometry.size() * monitor->m_scale).round();
    if (contentPx.x < 1 || contentPx.y < 1 || contentPx.x > monitor->m_pixelSize.x || contentPx.y > monitor->m_pixelSize.y) {
        m_failure = "content larger than monitor";
        return false;
    }

    // Monitor-sized so beginFullFakeRender's projection applies unchanged; content sits at the top-left corner.
    if (!out.fb)
        out.fb = g_pHyprRenderer->createFB("resize-transition snapshot");
    if (out.fb->m_size != monitor->m_pixelSize && !out.fb->alloc(monitor->m_pixelSize.x, monitor->m_pixelSize.y, DRM_FORMAT_ABGR8888)) {
        m_failure = "framebuffer alloc failed";
        return false;
    }
    out.fb->setImageDescription(monitor->workBufferImageDescription());

    CRegion fullDamage{0, 0, monitor->m_transformedSize.x, monitor->m_transformedSize.y};
    if (!g_pHyprRenderer->beginFullFakeRender(monitor, fullDamage, out.fb)) {
        m_failure = "beginFullFakeRender failed";
        return false;
    }
    g_pHyprRenderer->draw(CClearPassElement::SClearData{CHyprColor(0, 0, 0, 0)});

    bool texturesOk = true;
    surface->breadthfirst(
        [&](SP<CWLSurfaceResource> s, const Vector2D& offset, void*) {
            const auto& state = s->m_current;
            if (!state.texture || state.size.x < 1 || state.size.y < 1)
                return;
            if (!state.texture->ok()) {
                texturesOk = false;
                return;
            }
            // Same viewport-source UV as IElementRenderer::calculateUVForSurface, minus its window-size stretch/crop (the F12 cause).
            if (state.viewport.hasSource && state.bufferSize.x > 0 && state.bufferSize.y > 0) {
                const auto& src = state.viewport.source;
                setCustomUV(src.pos() / state.bufferSize, (src.pos() + src.size()) / state.bufferSize);
            } else
                setCustomUV(UV_UNSET, UV_UNSET);
            CBox box = CBox{offset - geometry.pos(), state.size}.scale(monitor->m_scale).round();
            CTexPassElement::SRenderData data;
            data.tex           = state.texture;
            data.box           = box;
            data.allowCustomUV = true;
            data.surface       = s;
            g_pHyprRenderer->draw(data, CRegion{box});
        },
        nullptr);
    setCustomUV(UV_UNSET, UV_UNSET);
    g_pHyprRenderer->endRender();

    if (!texturesOk) {
        m_failure = "surface texture not ok";
        return false;
    }
    out.contentPx = contentPx;
    return true;
}

bool CResizeTransition::beginGeneration(const Vector2D& goalSize) {
    // Retarget mid-transition: committed content (old, or the previous generation's new buffer) becomes the new old.
    if (!capture(m_old))
        return false;
    m_targetAck = goalSize.clamp(Vector2D{0, 0}, Math::VECTOR2D_MAX).floor();
    m_newReady  = false;
    m_switching = false;
    m_fade      = 0.F;
    m_animEnded = false;
    m_mesh.reset();
    m_moving    = false;
    m_revoked   = false;
    m_beginSize = m_window->sizeAnimation()->value();
    m_goalSize  = goalSize;
    m_progress  = 0;
    return true;
}

bool CResizeTransition::offerMesh(SMesh&& mesh) {
    if (m_moving || mesh.leaves.empty())
        return false;
    m_mesh    = std::move(mesh);
    m_revoked = false;
    return true;
}

bool CResizeTransition::revokeMesh() {
    if (!m_mesh || m_revoked)
        return false;
    m_revoked   = true;
    m_revokedAt = steady_clock::now();
    return true;
}

// Presenter: old and new content through the mesh (or the degenerate one) at m_progress, summed by the fade, into m_composed.
bool CResizeTransition::compose(const Vector2D& boxSize, float weight) {
    const auto window  = m_window.lock();
    const auto monitor = window->m_monitor.lock();
    if (!monitor) {
        m_failure = "no monitor";
        return false;
    }
    if (!m_composed.fb)
        m_composed.fb = g_pHyprRenderer->createFB("resize-transition composed");
    if (m_composed.fb->m_size != monitor->m_pixelSize && !m_composed.fb->alloc(monitor->m_pixelSize.x, monitor->m_pixelSize.y, DRM_FORMAT_ABGR8888)) {
        m_failure = "framebuffer alloc failed";
        return false;
    }
    m_composed.fb->setImageDescription(monitor->workBufferImageDescription());
    auto* const target = dynamic_cast<Render::GL::CGLFramebuffer*>(m_composed.fb.get());
    if (!target) {
        m_failure = "renderer is not OpenGL";
        return false;
    }
    m_composed.contentPx = (boxSize * monitor->m_scale).round().clamp(Vector2D{1, 1}, monitor->m_pixelSize);

    const bool showNew = m_newReady && m_new.fb;
    // weight 0 is the degenerate mesh whatever the leaves are: one whole-content leaf says the same with two triangles.
    const bool               meshed = m_mesh && weight > 0.F;
    const Vector2D           newPx  = showNew ? m_new.contentPx : m_old.contentPx;
    const Geometry::SSize    oldCanvas = meshed ? m_mesh->oldCanvas : Geometry::SSize{m_old.contentPx.x, m_old.contentPx.y};
    const Geometry::SSize    newCanvas = meshed ? m_mesh->newCanvas : Geometry::SSize{newPx.x, newPx.y};
    std::vector<prep::Leaf>  whole;
    if (!meshed) {
        prep::Leaf leaf;
        leaf.oldBox      = {0, 0, sc<int>(oldCanvas.w), sc<int>(oldCanvas.h)};
        leaf.newBox      = {0, 0, sc<int>(newCanvas.w), sc<int>(newCanvas.h)};
        leaf.midpointBox = {0, 0, (oldCanvas.w + newCanvas.w) / 2, (oldCanvas.h + newCanvas.h) / 2};
        whole.push_back(std::move(leaf));
    }
    const auto& leaves = meshed ? m_mesh->leaves : whole;

    Presenter::SLayer layers[2];
    size_t            count = 0;
    const auto        add   = [&](Geometry::ESide side, const SSnapshot& snap, float opacity) {
        if (opacity <= 0.F)
            return;
        auto& batch = m_batches[side];
        batch.vertices.clear();
        batch.indices.clear();
        Geometry::build(leaves, oldCanvas, newCanvas, side, m_progress, meshed ? weight : 0., batch);
        layers[count++] = {.texture = snap.fb->getTexture()->m_texID,
                           .uExtent = sc<float>(snap.contentPx.x / snap.fb->m_size.x),
                           .vExtent = sc<float>(snap.contentPx.y / snap.fb->m_size.y),
                           .batch   = &batch,
                           .opacity = opacity};
    };
    add(Geometry::OLD, m_old, showNew ? 1.F - m_fade : 1.F);
    if (showNew)
        add(Geometry::NEW, m_new, m_fade);
    if (!Presenter::compose(target->getFBID(), sc<int>(m_composed.contentPx.x), sc<int>(m_composed.contentPx.y), std::span{layers, count})) {
        m_failure = Presenter::failure();
        return false;
    }
    return true;
}

bool CResizeTransition::prepareFrame(const STransitionConfig& config, bool held) {
    const auto window = m_window.lock();
    if (!window || !window->m_isMapped) {
        m_failure = "window gone";
        return false;
    }
    const auto now = steady_clock::now();

    // ackedSize lands in m_current with the first commit after ack_configure (CWindow::onAck). That buffer can still be a client-side
    // transitional frame (kitty: old cells at top-left, rest blank) -- readiness judges it from the captured m_new, also during a hold.
    if (!m_newReady) {
        const auto surface = window->resource();
        if (surface && surface->m_current.ackedSize == m_targetAck) {
            m_newReady = true;
            Log::logger->log(Log::DEBUG, "[resize-transition] new buffer: ack {} surface {} geometry {}", m_targetAck, surface->m_current.size,
                             window->m_xdgSurface->m_current.geometry.size());
        }
    }
    if (m_newReady) {
        // Re-captured every frame: desynced subsurfaces commit without the main surface, so commit-driven capture would miss them.
        if (!capture(m_new))
            return false;
    }

    if (!held)
        m_moving = true;
    // Geometry t follows the box itself (D3-2), along the axis that changes most: held -> 0, animation over -> 1.
    const auto     size  = window->sizeAnimation()->value();
    const Vector2D delta = m_goalSize - m_beginSize;
    const bool     wide  = std::abs(delta.x) >= std::abs(delta.y);
    const double   span  = wide ? delta.x : delta.y;
    m_progress           = span == 0 ? 1. : std::clamp((wide ? size.x - m_beginSize.x : size.y - m_beginSize.y) / span, 0., 1.);
    float weight         = 0.F;
    if (m_mesh)
        weight = !m_revoked ? 1.F : config.meshRampMs <= 0 ? 0.F : std::clamp(1.F - sc<float>(msSince(m_revokedAt, now) / config.meshRampMs), 0.F, 1.F);

    if (m_newReady) {
        // Old -> new switch: with a mesh, at the midpoint, where the two sides' geometry agrees; plain scale has no such point and
        // switches as soon as the box moves.
        if (!held && !m_switching && (weight <= 0.F || m_progress >= .5)) {
            m_switching = true;
            m_switchAt  = now;
        }
        if (m_switching)
            m_fade = config.crossfadeMs <= 0 ? 1.F : std::clamp(sc<float>(msSince(m_switchAt, now) / config.crossfadeMs), 0.F, 1.F);
    }
    if (!compose(size, weight))
        return false;

    const bool animating = window->sizeAnimation()->isBeingAnimated();
    if (animating)
        m_animEnded = false;
    else if (!m_animEnded) {
        m_animEnded = true;
        m_animEndAt = now;
    }

    if (!animating && m_newReady && m_fade >= 1.F) {
        m_failure.clear();
        return false; // done
    }
    if (!animating && !m_newReady && msSince(m_animEndAt, now) > config.newBufferTimeoutMs) {
        m_failure = "client did not commit the new size in time";
        return false;
    }
    return true;
}

void CResizeTransition::preWindowRender(CSurfacePassElement::SRenderData* renderData) {
    m_frameValid = false;
    if (!renderData || !m_composed.fb)
        return;
    const auto monitor = renderData->pMonitor.lock();
    if (!monitor)
        return;

    // Same box the native path uses for the main surface: animated position/size (+ workspace/floating offsets already in pos).
    m_contentBox = CBox{renderData->pos - monitor->m_position, {renderData->w, renderData->h}}.scale(monitor->m_scale).round();
    m_drawAlpha  = renderData->alpha * renderData->fadeAlpha;
    // "- 1" mirrors IElementRenderer::drawSurface (keeps the border's inner edge covered).
    m_round      = renderData->dontRound ? 0 : std::max(renderData->rounding - 1, 0);
    m_roundPower = renderData->dontRound ? 2.F : renderData->roundingPower;
    m_frameValid = true;

    // Native surfaces still get pass elements but draw invisibly; decorations already captured fullAlpha before this hook.
    // Restored in amendTransformedRenderData so popups (rendered later from the same SRenderData) stay visible.
    m_pendingRenderData = renderData;
    m_savedAlpha        = renderData->alpha;
    renderData->alpha   = 0.F;
}

void CResizeTransition::amendTransformedRenderData(const CBox&, SMotionBlurData*) {
    if (!m_pendingRenderData)
        return;
    m_pendingRenderData->alpha = m_savedAlpha;
    m_pendingRenderData        = nullptr;
}

void CResizeTransition::drawSnapshot(const SSnapshot& snap, const CBox& box, float alpha) {
    if (!snap.fb || alpha <= 0.F)
        return;
    setCustomUV({0, 0}, snap.contentPx / snap.fb->m_size);
    CTexPassElement::SRenderData data;
    data.tex           = snap.fb->getTexture();
    data.box           = box;
    data.a             = alpha;
    data.round         = m_round;
    data.roundingPower = m_roundPower;
    data.allowCustomUV = true;
    g_pHyprRenderer->draw(data, CRegion{box});
    setCustomUV(UV_UNSET, UV_UNSET);
}

// The composed image, scaled to the animated content box (1:1 but for sub-pixel rounding). `in` is already bound by drawTransformedWindow.
SP<Render::IFramebuffer> CResizeTransition::transform(SP<Render::IFramebuffer> in) {
    if (!m_frameValid)
        return in;
    m_frameValid = false;

    drawSnapshot(m_composed, m_contentBox, m_drawAlpha);
    return in;
}
