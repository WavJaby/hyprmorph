#pragma once
// The S1-B + D4-1 preparation chain (port of acceptance_pipeline.run_chain + joint_alignment_probe.select_partition):
// ORB -> proposals -> pairing -> baseline + joint partitions on the default midpoint, choose one -> midpoint relayout
// (F1 vanishing collapse) + local fits -> rigid strips -> rigid patches -> flat absorb -> point constraints.
#include "cg.hpp"
#include "gles.hpp"
#include "json.hpp"
#include "partition.hpp"
#include "rigid.hpp"
#include "strip_ncc.hpp"
#include "types.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace prep {

struct Config {
    int maxRegions = 20;           // partition leaf budget (both partitions)
    bool collapseVanishing = true;  // F1 rule
    std::vector<std::string> stages{"rigid", "flat", "constraints"};
};

struct Result {
    std::vector<Leaf> leaves;
    std::vector<Split> splits;
    std::vector<std::pair<std::string, double>> timings;  // stage -> ms, in run order
    json::Value evidence = json::Value::object();
    double prepMs() const;
};

class Preparer {
public:
    explicit Preparer(gles::Context& context);
    // RANSAC draws from OpenCV's thread RNG: seeded here per run, so a case's result never depends on earlier runs.
    static constexpr uint64_t RNG_SEED = 20261019;
    Result run(const cv::Mat& oldImage, const cv::Mat& newImage, const Config& config);
    // First dispatch of each kernel pays the driver's shader JIT, and OpenCV pays first-use costs (measured: a cold
    // first request ~430-510 ms vs ~85 ms warm): run both once on synthetic data at startup. Returns elapsed ms.
    double warm();

private:
    StripNcc ncc;
    CgSolver cg;
};

} // namespace prep
