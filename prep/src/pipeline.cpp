#include "pipeline.hpp"

#include "clip.hpp"
#include "constraints.hpp"
#include "fit.hpp"
#include "flat.hpp"
#include "pairing.hpp"
#include "proposals.hpp"

#include <opencv2/core/utility.hpp>

#include <cstdlib>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <future>

namespace prep {

double Result::prepMs() const {
    double total = 0;
    for (auto& [k, v] : timings) total += v;
    return total;
}

namespace {
// No stage uses cv::UMat, but OpenCV initialises its OpenCL runtime on first use anyway: every installed ICD is loaded (NVIDIA OpenCL +
// CUDA, Intel) and /dev/nvidia* is opened -- ~140 MB of PSS and a discrete-GPU touch for nothing; GPU work here is GLES.
// cv::ocl::setUseOpenCL(false) does not stop the load; the loader only honours this variable, read at its first use.
gles::Context& withoutOpenCL(gles::Context& context) {
    setenv("OPENCV_OPENCL_RUNTIME", "disabled", 0);
    return context;
}
}

Preparer::Preparer(gles::Context& context) : ncc(withoutOpenCL(context)), cg(context) {}

double Preparer::warm() {
    auto started = std::chrono::steady_clock::now();
    // A UI-like pair (top bar strip, side panel, cards, text-like noise) resized like a tiled -> floating window, so
    // every stage and both GPU kernels run at realistic sizes; then each kernel once more on its own in case the
    // synthetic pair skipped it.
    cv::Mat a(420, 700, CV_8UC3, cv::Scalar(245, 245, 245));
    cv::RNG rng(1);
    cv::rectangle(a, cv::Rect(0, 0, 700, 40), cv::Scalar(60, 60, 60), cv::FILLED);
    cv::rectangle(a, cv::Rect(0, 40, 160, 380), cv::Scalar(225, 225, 225), cv::FILLED);
    for (int i = 0; i < 12; ++i) cv::rectangle(a, cv::Rect(12 + 52 * i, 10, 20, 20), cv::Scalar(200, 120 + 8 * i, 40), cv::FILLED);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            cv::Rect card(190 + 165 * c, 70 + 115 * r, 140, 95);
            cv::rectangle(a, card, cv::Scalar(255, 255, 255), cv::FILLED);
            cv::rectangle(a, card, cv::Scalar(180, 180, 180), 1);
            cv::Mat text = a(cv::Rect(card.x + 10, card.y + 60, 110, 20));
            rng.fill(text, cv::RNG::UNIFORM, 0, 256);
        }
    cv::Mat b;
    cv::resize(a, b, cv::Size(390, 385), 0, 0, cv::INTER_AREA);
    run(a, b, Config{});
    cv::Mat stripA = a(cv::Rect(0, 5, 700, 30)).clone(), stripB = b(cv::Rect(0, 5, 390, 30)).clone();
    ncc.score({{stripA, stripB}});
    SparseSystem s;
    s.cols = 2;
    s.add(0, 1.);
    s.endRow(1.);
    s.add(1, 1.);
    s.endRow(2.);
    cg.solve({s});
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

Result Preparer::run(const cv::Mat& oldImage, const cv::Mat& newImage, const Config& config) {
    using Clock = std::chrono::steady_clock;
    Result out;
    auto started = Clock::now();
    auto lap = [&](const char* name) {
        auto now = Clock::now();
        out.timings.emplace_back(name, std::chrono::duration<double, std::milli>(now - started).count());
        started = now;
    };
    cv::setRNGSeed(int(RNG_SEED));
    Shape oldShape{oldImage.cols, oldImage.rows}, newShape{newImage.cols, newImage.rows};

    Points source, target;
    matchImages(oldImage, newImage, source, target);
    lap("matching");

    // Per-image work with no shared state runs the old image on a second thread (results identical by construction;
    // none of it draws from the RNG, which only RANSAC on this thread uses).
    auto pendingProposals = std::async(std::launch::async, [&] { return propose(oldImage); });
    ImageProposals newProposals = propose(newImage), oldProposals = pendingProposals.get();
    lap("proposals");

    std::vector<Pair> coarse = selectSafeCandidates(pairRegions(oldProposals.regions, newProposals.regions, oldShape, newShape, source, target),
                                                    oldShape, newShape);
    std::vector<Pair> small;
    for (auto& p : chooseSupported(oldProposals.small, newProposals.small, source, target, oldShape, newShape))
        if (!overlapsCoarse(p, coarse)) small.push_back(p);
    std::vector<Pair> both = coarse;
    both.insert(both.end(), small.begin(), small.end());
    Points trustedSource, trustedTarget;
    verifiedLandmarks(source, target, both, trustedSource, trustedTarget);
    lap("pairing");

    auto pendingSeams = std::async(std::launch::async, [&] { return SeamIndex(oldImage); });
    SeamIndex newSeams(newImage), oldSeams = pendingSeams.get();
    GutterFinder gutters(oldImage, newImage, source, target);
    SeamCache seams{oldSeams, newSeams, gutters, {}};
    PartitionResult baseline = baselinePartition(oldShape, newShape, coarse, config.maxDepth, seams);
    PartitionResult joint;
    std::string selected = "baseline", reason = "joint_disabled";
    if (config.partition == PartitionChoice::Preview) {
        joint = jointPartition(oldShape, newShape, coarse, small, trustedSource, trustedTarget, .7, config.maxDepth, seams);
        // The partition is chosen on the default (average) midpoint layout: a layout policy must not pick the structure.
        std::vector<ContourDetail> baseDetails, jointDetails;
        auto baseContour = contourMidpointError(baseline.leaves, coarse, baseDetails);
        auto jointContour = contourMidpointError(joint.leaves, coarse, jointDetails);
        auto baseLandmark = midpointError(baseline.leaves, trustedSource, trustedTarget);
        auto jointLandmark = midpointError(joint.leaves, trustedSource, trustedTarget);
        std::tie(selected, reason) = choosePreview(baseContour, jointContour, baseLandmark, jointLandmark, baseDetails, jointDetails);
    }
    PartitionResult& chosen = selected == "joint" ? joint : baseline;
    lap("partition");

    // Clip containers join the tree before the midpoint layout (it sees one leaf per container), then leave the leaf list: the fits and
    // the deforming stages below never touch them. They are appended to the result unchanged.
    std::vector<std::string> clipNames = mergeClipContainers(oldImage, newImage, chosen);
    lap("clip_containers");
    std::map<std::string, int> collapse;
    if (config.collapseVanishing) collapse = vanishingCollapses(chosen.leaves, chosen.splits, source, target);
    std::vector<std::string> relaid = relayout(chosen.leaves, chosen.splits, collapse);
    std::vector<Leaf> clips;
    for (auto& leaf : chosen.leaves)
        if (leaf.clip) clips.push_back(leaf);
    std::erase_if(chosen.leaves, [](const Leaf& leaf) { return leaf.clip.has_value(); });
    std::vector<Pair> refine = coarse;
    for (auto& p : small)
        if (std::any_of(chosen.leaves.begin(), chosen.leaves.end(), [&](const Leaf& l) { return containedPair(p, l); })) refine.push_back(p);
    refineRegions(oldImage, newImage, chosen.leaves, refine);
    lap("fits");
    out.leaves = std::move(chosen.leaves);
    out.splits = std::move(chosen.splits);
    {
        json::Array relaidJson;
        for (auto& n : relaid) relaidJson.push_back(n);
        out.evidence.set("selected_partition", selected);
        out.evidence.set("selection_reason", reason);
        out.evidence.set("regions", int(out.leaves.size() + clips.size()));
        json::Array clipJson;
        for (auto& n : clipNames) clipJson.push_back(n);
        out.evidence.set("clip_containers", clipJson);
        out.evidence.set("max_depth", config.maxDepth);
        out.evidence.set("depth_limited", chosen.depthLimited);
        out.evidence.set("paired_cuts", int(out.splits.size()));
        out.evidence.set("matches", int(source.size()));
        out.evidence.set("verified_landmarks", int(trustedSource.size()));
        out.evidence.set("midpoint_relayout", relaidJson);
    }

    RigidStage rigid(ncc);
    for (auto& stage : config.stages) {
        if (stage == "rigid") {
            rigid.prepare(oldImage, newImage, out.leaves);
            lap("rigid_scores");
            if (config.gpuPatches) {
                out.evidence.set("gpu_patch_strips_accepted", rigid.patches(oldImage, newImage, out.leaves));
                lap("gpu_patches");
            }
        } else if (stage == "flat") {
            auto results = applyFlatAbsorb(oldImage, newImage, out.leaves, source, target);
            lap("flat_absorb");
            int accepted = 0;
            json::Array unabsorbed, background;
            for (auto& r : results) {
                accepted += r.accepted;
                if (!r.unabsorbed.empty()) unabsorbed.push_back(r.name);
                for (auto& b : r.background) background.push_back(r.name + ":" + b);
            }
            out.evidence.set("flat_absorb_accepted", accepted);
            out.evidence.set("flat_unabsorbed", unabsorbed);
            out.evidence.set("flat_background", background);
        } else if (stage == "constraints") {
            auto results = applyPointConstraints(oldImage, newImage, out.leaves, source, target, cg);
            lap("point_constraints");
            int accepted = 0;
            for (auto& r : results) accepted += r.accepted;
            out.evidence.set("constraint_regions_accepted", accepted);
        } else
            throw std::runtime_error("unknown stage " + stage);
    }
    out.leaves.insert(out.leaves.end(), std::make_move_iterator(clips.begin()), std::make_move_iterator(clips.end()));
    return out;
}

} // namespace prep
