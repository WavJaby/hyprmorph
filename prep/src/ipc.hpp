#pragma once
// Prep worker IPC, OpenCV side: cv::Mat <-> image memfd and the blocking Client the tools use. Wire format and transport: wire.hpp.
#include "types.hpp"
#include "wire.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace prep::ipc {

// Copies a BGR8 image into a sealed memfd; returns the fd (caller closes).
int imageMemfd(const cv::Mat& bgr, ImageDesc& desc);
// Maps an image memfd read-only and converts it to a CV_8UC3 BGR Mat (copy); throws on a malformed descriptor.
cv::Mat readImage(int fd, const ImageDesc& desc);

struct Response {
    uint64_t generation = 0;
    std::string refusal;  // empty when status == Ok
    std::vector<Leaf> leaves;
    std::vector<std::pair<std::string, double>> timings;
    json::Value evidence;
    double roundTripMs = 0;  // client-measured: send -> payload parsed
};

class Client {
public:
    explicit Client(const std::string& path = socketPath());
    ~Client();
    Response prepare(uint64_t generation, const cv::Mat& oldImage, const cv::Mat& newImage, uint64_t window = 0);

private:
    int fd = -1;
};

} // namespace prep::ipc
