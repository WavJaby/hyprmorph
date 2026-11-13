// Per-window resize transition: old snapshot + live new content, composed by one Presenter path -- a prepared mesh when one was offered in
// time, the plain-scale degenerate mesh otherwise -- into a box-sized image that is then drawn like a surface (rounding, alpha).
// Owned by the window's transformer list (CWindow::effects().transformers()) while a generation is active; detached (-> native drawing) when it ends or on any error.
#pragma once

#include "Geometry.hpp"

#include <hyprland/src/render/transformer/Transformer.hpp>
#include <hyprland/src/render/WindowRenderPresentation.hpp>
#include <hyprland/src/render/Framebuffer.hpp>
#include <hyprland/src/desktop/DesktopTypes.hpp>

#include <chrono>
#include <optional>
#include <string>
#include <vector>

struct STransitionConfig {
    double crossfadeSpan;      // old -> new fade, in transition progress, from the t = 0.5 switch
    long   crossfadeMs;        // fade length when the new-size buffer arrives after that window opened
    long   newBufferTimeoutMs; // after the size animation ends, wait this long for the new-size buffer, then native
    long   meshRampMs;         // mesh weight ramp on a revoke
    double crossfadeBlurPx;    // peak Gaussian sigma, logical px, at the middle of the crossfade (0 = off)
};

// A prepared mesh: leaves in analysis px of the two images the prep worker was given.
struct SMesh {
    std::vector<prep::Leaf> leaves;
    Geometry::SSize         oldCanvas, newCanvas;
};

class CResizeTransition : public Render::IWindowTransformer {
  public:
    explicit CResizeTransition(PHLWINDOWREF window);
    ~CResizeTransition() override;

    // Starts (or retargets) a generation: captures the currently committed content as the old snapshot. false -> caller detaches.
    bool                     beginGeneration(const Vector2D& goalSize);

    // Called before each frame of the window's monitor. false -> transition is over (done or failed); caller detaches.
    // held: the animation is held -> the box still has its begin size, so the old snapshot stays; the switch waits for release.
    bool                     prepareFrame(const STransitionConfig& config, bool held);

    // Offers this generation's mesh. Taken while only old content is on screen and the box is short of the crossfade window: transition
    // progress then restarts at 0 from the box's current progress (the mesh's t = 0 is the plain scale already shown, so it joins without a
    // jump). false -> too late, the generation stays plain scale. joined: taken after the box started moving.
    bool                     offerMesh(SMesh&& mesh, bool& joined);
    // The selection the mesh came from was revoked -> its weight ramps to 0 over meshRampMs, leaving plain scale.
    // true -> a drawn mesh started its ramp out.
    bool                     revokeMesh();

    // Removal from the window's transformer list goes through Render::CWindowTransformerList::removeInactive (it has no erase by
    // pointer): detach() marks the transition inactive, then prunes.
    void                     deactivate() {
        m_active = false;
    }
    bool                             active() const override {
        return m_active;
    }
    Render::SWindowTransformBuffer   transform(Render::CRenderContext& ctx, const Render::SWindowTransformBuffer& in, const Render::SWindowTransformContext& context) override;
    void                             preWindowRender(Render::CRenderContext& ctx, CSurfacePassElement::SRenderData* renderData) override;
    void                             amendTransformedRenderData(Render::CRenderContext& ctx, const CBox& currentBox, SMotionBlurData* motionBlur,
                                                                const Render::SWindowRenderPresentation& presentation) override;

    std::string              m_failure; // last reason the transition gave up, for the log

    struct SSnapshot {
        SP<Render::IFramebuffer> fb;
        Vector2D                 contentPx; // content occupies (0,0)-contentPx of fb, in fb pixels
    };
    // This frame's capture of the client's new-size content, or null before the client acked the target size. Readiness samples it.
    const SSnapshot* newContent() const {
        return m_newReady && m_new.fb ? &m_new : nullptr;
    }
    // The generation's old content (captured at beginGeneration). Readiness builds its placeholder alias from it.
    const SSnapshot& oldContent() const {
        return m_old;
    }

  private:

    bool                             capture(SSnapshot& out);
    bool                             compose(const Vector2D& boxSize, float weight, double blurPx);
    void                             drawSnapshot(Render::CRenderContext& ctx, const SSnapshot& snap, const CBox& box, float alpha);

    PHLWINDOWREF                     m_window;
    bool                             m_active = true;
    SSnapshot                        m_old, m_new;
    SSnapshot                        m_composed; // this frame's Presenter output, contentPx = the animated box in device px
    SP<Render::IFramebuffer>         m_blurPass; // horizontal blur pass of m_composed, read back by the vertical one
    std::optional<SMesh>             m_mesh;
    bool                             m_moving  = false; // the box has left its begin size (first frame not held)
    bool                             m_revoked = false;
    std::chrono::steady_clock::time_point m_revokedAt;
    Vector2D                         m_beginSize, m_goalSize; // size animation value at beginGeneration, and its goal
    // Size follows Hyprland's animation; content follows transition progress, derived from it so both end together:
    // t = (box - join) / (1 - join), join = box progress when the mesh was taken (0 without a mesh, or with one taken during the hold).
    double                           m_boxProgress = 0; // the box's way from m_beginSize to m_goalSize on the axis that changes most, 0..1
    double                           m_join        = 0;
    double                           m_transition  = 0; // t: drives mesh geometry and the old -> new crossfade
    Geometry::SBatch                 m_batches[2];
    Vector2D                         m_targetAck; // configure size the client must ack (CWaylandBackend::configure records the configured size floored)
    bool                             m_newReady    = false; // client acked the target size; m_new is captured every frame from here
    bool                             m_windowOpen  = false; // t reached the crossfade window (with or without new content)
    bool                             m_switching   = false; // old -> new fade started
    bool                             m_fadeByClock = false; // new content arrived after the window opened: fade over crossfadeMs from then
    std::chrono::steady_clock::time_point m_switchAt, m_animEndAt;
    bool                             m_animEnded = false;
    float                            m_fade      = 0.F; // 0 = old only, 1 = new only

    // Valid only between preWindowRender and amendTransformedRenderData of one renderWindow call: points at that call's local SRenderData.
    // Pass elements copy SRenderData when created, so restoring alpha there leaves the main surfaces' elements at 0 and popups' visible.
    CSurfacePassElement::SRenderData* m_pendingRenderData = nullptr;
    float                             m_savedAlpha        = 1.F;

    // Frame inputs saved in preWindowRender, consumed by transform() later in the same frame.
    CBox                             m_contentBox; // monitor-local, scaled pixels; transform() draws it into the canvas at minus the canvas origin
    float                            m_drawAlpha = 1.F;
    int                              m_round     = 0;
    float                            m_roundPower = 2.F;
    bool                             m_frameValid = false;
};
