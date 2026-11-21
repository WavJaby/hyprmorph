#pragma once
// Rigid strips + whole-footprint rigid patches.
//
// Both the "strip" stage and the "patch" stage run the same GPU NCC matcher on the same crops (the stages were tuned
// that way). It is computed once per strip leaf (one GPU batch per case) and reused by
// both stages -- identical input, identical result, half the NCC work.
#include "json.hpp"
#include "strip_ncc.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <map>
#include <optional>
#include <vector>

namespace prep {

struct Landmark {
    double oldX = 0, newX = 0, oldY = 0, newY = 0, score = 0;
    int width = 0, height = 0;
    bool grown = false;  // complete_foreground expansion accepted
    double seedOldX = 0, seedNewX = 0;
    int seedWidth = 0;
    json::Value toJson() const;
};

// Strip matcher, CPU part, given the GPU scores of (a, b).
std::vector<Landmark> stripLandmarks(const cv::Mat& a, const cv::Mat& b, const StripScores& scores);

struct StripMesh {
    std::optional<Meshes> mesh;
    json::Value result;  // accepted / reason / landmarks ... (rigid_constraints without the windows)
    std::vector<std::array<int, 4>> windows;
};

// Strip mesh from precomputed landmarks of the (possibly transposed) strip.
StripMesh meshForStrip(const cv::Mat& oldPatch, const cv::Mat& newPatch, bool vertical, const double middle[2],
                       const std::vector<Landmark>& landmarks);

enum class StripOrientation { None, Horizontal, Vertical };
StripOrientation stripOrientation(const Leaf& leaf);

class RigidStage {
public:
    explicit RigidStage(StripNcc& ncc) : ncc(ncc) {}
    // Scores every strip leaf once (one GPU batch); must run before patches() of the same leaves.
    void prepare(const cv::Mat& oldImage, const cv::Mat& newImage, const std::vector<Leaf>& leaves);
    // apply_gpu_patches: meshes for every strip leaf, each gated by audit_leaf. Returns accepted count.
    int patches(const cv::Mat& oldImage, const cv::Mat& newImage, std::vector<Leaf>& leaves);
    json::Value report = json::Value::array();

private:
    StripNcc& ncc;
    std::map<int, StripMesh> meshes;  // by leaf index
};

} // namespace prep
