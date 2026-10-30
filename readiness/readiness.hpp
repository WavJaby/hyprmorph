// Layout readiness detector: decides when a window's new-size frames have settled enough to build the transition mesh.
// C++ port of StructureReadiness / RegionStructureReadiness (experiments/frame_readiness.py, region_layout_readiness.py); the Python code stays
// the parity reference (D1-5). Contract (ReadinessDetector seam): observe(frame) -> waiting / ready(candidate) / revoked.
// No compositor or OpenCV dependency; input is the sampler's GPU thumbnail (already downscaled to the analysis width) as 8-bit pixels.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace readiness {

// 8-bit interleaved pixels; only the first 3 channels are read (channel order is irrelevant: every metric takes a max over channels).
struct Image {
    int width = 0, height = 0, channels = 3;
    size_t stride = 0;  // bytes per row
    const uint8_t* data = nullptr;
};

// What revokes a selected frame (F8/F9).
enum class PostAcceptance {
    Strict,        // Python reference: any non-quiet frame revokes, so an animated region re-triggers selection
    LayoutChange,  // only a large contour change against the selected frame revokes; smaller changes are content (animation, cursor, scrollbar)
};

enum class QuietRule {
    WorstTile,      // StructureReadiness: global edge diff + every informative cell under the cell limit
    StableRegions,  // RegionStructureReadiness: global edge diff + a stable majority of informative cells (no worst-cell veto)
};

// Every threshold of the detector. Defaults = the adopted layout-amplitude rule (cross-app readiness, RegionStructurePolicy).
struct Policy {
    QuietRule quietRule = QuietRule::StableRegions;
    // Timing. Confirm window: the frame must stay quiet this long, over at least this many observations, before it is selected.
    double confirmMs = 60.;                // quiet_ms; 60 ms replayed with 0 later large changes on 90 decidable trials
    int minimumConfirmObservations = 3;    // minimum_quiet_samples
    double maximumSampleGapMs = 60.;       // a longer gap between observations restarts the confirm window
    double maximumWaitMs = 500.;           // no selection after this; caller falls back to plain scale
    // "Must have left old stretch/crop": the frame has to differ from every old-image alias by both measures.
    double aliasPixelThreshold = 4.;       // per-pixel max-channel difference (0-255 scale) counted as changed
    double updateMeanDifference = .45;     // update_mae, 0-255 scale
    double updateChangedFraction = .01;    // update_fraction
    // Structure features: Gaussian blur -> Sobel H/V -> long-edge filter.
    double blurSigma = 1.2;                // low-pass before gradients; suppresses antialiasing and text noise
    double gradientThreshold = .8;         // Sobel magnitude (scale 1/8, 0-255 input) for an edge pixel
    double gradientDominance = 2.;         // vertical edge needs |gx| > dominance*|gy| (and vice versa)
    double minimumLineFraction = .10;      // opening length as fraction of window height (vertical) / width (horizontal)
    int minimumLinePixels = 9;             // opening length floor, analysis pixels
    int borderPixels = 2;                  // edge pixels cleared at the frame border (Sobel border artefacts)
    int minimumEdgePixels = 32;            // fewer edge pixels in total -> structure unsupported, never ready
    // Comparison: 1 px tolerant edge matching on a fixed grid.
    int edgeTolerancePixels = 1;           // an edge matches if the other frame has one within this radius (3x3 for 1)
    int gridRows = 8, gridColumns = 12;    // contour cells
    int minimumCellSupport = 12;           // edge pixels (both frames) for a cell to be informative
    double edgeDisagreement = .04;         // global unmatched-edge fraction for quiet
    double cellDisagreement = .30;         // per-cell unmatched fraction: limit (WorstTile) / stable cell (StableRegions)
    double minimumStableCells = .80;       // StableRegions: fraction of informative cells that must be stable
    // F7 static layout: a client commit at the target surface size, acked after this resize, is update evidence by itself. The aliases model the
    // compositor presenting the old buffer at the new geometry; a new-size client buffer cannot be that, so a layout that legitimately does not
    // move (terminal text anchored top-left) is no longer mistaken for the old crop.
    bool protocolUpdateEvidence = true;
    PostAcceptance postAcceptance = PostAcceptance::LayoutChange;
    // F9: unmatched-edge fraction against the selected frame that revokes it. Derived by experiments/readiness_rules_gate.py from 250 recorded
    // traces: post-acceptance changes under the adopted rule reach at most 4.0 % (Discord animated profile background); new-size frames vs
    // the settled frame leave no value in 12.1-24.9 % above that; .174 = geometric midpoint of that empty interval. The gate fails if they drift.
    double revokeEdgeDisagreement = .174;
    // D5-1 placeholder: a protocol-updated frame that still matches an old-image alias (old content at its old place) does not start the
    // confirm window. 0 = rule off. Inert without aliases. Set by readiness_rules_gate.py --placeholder-sweep on 330 recorded resizes:
    // wait 350 = vesktop's longest placeholder life (~276 ms) + ~25 %; no recorded trial reached it, so the static (anchored) path is ungated.
    // After a placeholder the next frame confirms over 20 ms: 0 / 20 / 40 / 60 gave vesktop 237 / 245 / 267 / 287 ms and Chrome 93 / 96 /
    // 114 / 132 ms, all with 0 revocations and 0 accepted-then-changed; 20 keeps 8 ms of margin over 0.
    double placeholderWaitMs = 350.;
    double placeholderConfirmMs = 20.;
    // Placeholder = changed-pixel fraction against some old alias below this. Not the update_* pair (built for "has left the old image";
    // vesktop placeholders sit at 1.1-1.2 %, just above update_fraction). Derived on 2024 recorded new-size frames: frames a later frame
    // changes by >= revokeEdgeDisagreement reach at most 9.4 %, frames that stay start at 20.6 %; .139 = geometric midpoint of that gap.
    double placeholderChangedFraction = .139;

    // key=value overrides with the Python field names (quiet_ms, edge_disagreement, ...). Returns false on an unknown key or bad value.
    bool set(const std::string& key, const std::string& value);
    std::string describe() const;  // JSON object
};

// Long axis-aligned edges: two binary planes (vertical, horizontal), row-major.
struct Structure {
    int width = 0, height = 0;
    std::vector<uint8_t> vertical, horizontal;
    std::vector<uint8_t> verticalNear, horizontalNear;  // planes dilated by edgeTolerancePixels, computed once per frame for every compare
    int edgePixels() const;
};

struct Comparison {
    double edgeDisagreement = 0;     // unmatched edges both ways / all edges
    double worstCell = 0;            // max unmatched fraction over informative cells
    double stableCellFraction = 1;   // informative cells within cellDisagreement / informative cells
    int informativeCells = 0, changingCells = 0;
};

struct AliasDifference { double meanDifference = 0, changedFraction = 0; };

// Low-pass color image (float, 3 channels) used for alias tests and as the source of structure features.
struct Color {
    int width = 0, height = 0;
    std::vector<float> data;
};

// One old-image alias the new frame must differ from: old stretched to the target, or old clipped at the same coordinates (mask = shared interior).
struct Alias {
    Color color;
    std::vector<uint8_t> mask;  // empty = whole frame; else nonzero = pixel counted
};

enum class Decision { Waiting, Ready, Revoked };

// Per-frame protocol facts from the FrameSource adapter.
struct Evidence {
    bool targetGeometry = false;           // window/surface at the accepted target size
    bool newBuffer = false;                // client committed after acking this resize's configure (surface may snap to its own increments)
    std::optional<bool> recordedUpdate;    // replay only: original adapter's alias verdict, replaces the alias test
};

struct Observation {
    double elapsedMs = 0;
    bool supported = false, updated = false, updateEvidenceReused = false, targetGeometry = false, protocolUpdate = false;
    std::vector<AliasDifference> aliasDifferences;
    std::optional<Comparison> previous, anchor;
    std::optional<double> quietSinceMs;
    int quietSamples = 0;
    bool ready = false, newlyReady = false, revoked = false, expired = false;
    bool placeholder = false;  // D5-1: this frame matched an old-image alias while the rule was waiting for a different frame
    bool anchored = false;     // D5-1: the placeholder outlived the wait limit -> static layout, content stays in place, no mesh needed
    int generation = 0;  // candidate generation: bumps whenever the confirm window restarts
    std::optional<double> firstUpdateMs, readyMs;
    double analysisMs = 0;  // features + comparisons of this call, CPU wall time
    Decision decision() const { return revoked ? Decision::Revoked : ready ? Decision::Ready : Decision::Waiting; }
};

Color lowPass(const Image& image, const Policy& policy);
Structure structure(const Color& color, const Policy& policy);
Comparison compare(const Structure& a, const Structure& b, const Policy& policy);
AliasDifference aliasDifference(const Color& frame, const Alias& alias, const Policy& policy);
bool quiet(const Comparison& c, const Policy& policy);

// One detector per window per resize generation (D1-7): the caller creates a new one when the window's generation changes.
class Detector {
public:
    Detector(Policy policy, std::vector<Alias> aliases);
    Observation observe(const Image& frame, double elapsedMs, const Evidence& evidence);
    // Adapter veto of a just-selected frame (sample too old, or a resize arrived during analysis): undo the selection, keep the confirm window.
    void cancelSelection() { readyMs_.reset(); }
    const Policy& policy() const { return policy_; }

private:
    Policy policy_;
    std::vector<Alias> aliases_;
    std::optional<Structure> anchor_, previous_;
    std::optional<double> previousMs_, quietSinceMs_, updateMs_, readyMs_;
    int quietSamples_ = 0, generation_ = 0;
    bool updateEvidence_ = false;
    bool placeholderPhase_ = true, afterPlaceholder_ = false, anchored_ = false;  // D5-1
    std::optional<double> placeholderSinceMs_;
};

} // namespace readiness
