// Per-window resize transition: old snapshot + live new content, both drawn by one Presenter path (plain-scale degenerate mesh, D1-4).
// Owned by the window through CWindow::m_transformers while a generation is active; detached (-> native drawing) when it ends or on any error.
#pragma once

#include <hyprland/src/render/transformer/Transformer.hpp>
#include <hyprland/src/render/Framebuffer.hpp>
#include <hyprland/src/desktop/DesktopTypes.hpp>

#include <chrono>
#include <string>

struct STransitionConfig {
    long crossfadeMs;        // D3-1 old -> new fade length once the new-size buffer arrives
    long newBufferTimeoutMs; // after the size animation ends, wait this long for the new-size buffer, then native
};

class CResizeTransition : public Render::IWindowTransformer {
  public:
    explicit CResizeTransition(PHLWINDOWREF window);
    ~CResizeTransition() override;

    // Starts (or retargets) a generation: captures the currently committed content as the old snapshot. false -> caller detaches.
    bool                     beginGeneration(const Vector2D& goalSize);

    // Called before each frame of the window's monitor. false -> transition is over (done or failed); caller detaches.
    // held: the animation is held (D3-3) -> the box still has its begin size, so the old snapshot stays; the switch waits for release.
    bool                     prepareFrame(const STransitionConfig& config, bool held);

    SP<Render::IFramebuffer> transform(SP<Render::IFramebuffer> in) override;
    void                     preWindowRender(CSurfacePassElement::SRenderData* renderData) override;
    void                     amendTransformedRenderData(const CBox& currentBox, SMotionBlurData* motionBlur) override;

    std::string              m_failure; // last reason the transition gave up, for the log

    struct SSnapshot {
        SP<Render::IFramebuffer> fb;
        Vector2D                 contentPx; // content occupies (0,0)-contentPx of fb, in fb pixels
    };
    // This frame's capture of the client's new-size content, or null before the client acked the target size. Readiness samples it.
    const SSnapshot* newContent() const {
        return m_newReady && m_new.fb ? &m_new : nullptr;
    }
    // The generation's old content (captured at beginGeneration). Readiness builds its D5-1 placeholder alias from it.
    const SSnapshot& oldContent() const {
        return m_old;
    }

  private:

    bool                             capture(SSnapshot& out);
    void                             drawSnapshot(const SSnapshot& snap, const CBox& box, float alpha);

    PHLWINDOWREF                     m_window;
    SSnapshot                        m_old, m_new;
    Vector2D                         m_targetAck; // configure size the client must ack (CWindow::sendWindowSize stores goal.floor())
    bool                             m_newReady  = false; // client acked the target size; m_new is captured every frame from here
    bool                             m_switching = false; // old -> new fade running (starts at the first frame after any hold)
    std::chrono::steady_clock::time_point m_switchAt, m_animEndAt;
    bool                             m_animEnded = false;
    float                            m_fade      = 0.F; // 0 = old only, 1 = new only

    // Valid only between preWindowRender and amendTransformedRenderData of one renderWindow call: points at that call's local SRenderData.
    CSurfacePassElement::SRenderData* m_pendingRenderData = nullptr;
    float                             m_savedAlpha        = 1.F;

    // Frame inputs saved in preWindowRender, consumed by transform() later in the same frame.
    CBox                             m_contentBox; // monitor-local, scaled pixels
    float                            m_drawAlpha = 1.F;
    int                              m_round     = 0;
    float                            m_roundPower = 2.F;
    bool                             m_frameValid = false;
};
