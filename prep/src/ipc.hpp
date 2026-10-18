#pragma once
// Prep worker IPC contract (D1-1: preparation runs out of process, keyed by resize generation).
//
// Transport: one AF_UNIX SOCK_SEQPACKET connection per client, at $XDG_RUNTIME_DIR/resize-transition/prep.sock by
// default (socketPath()). Pixels never travel in the socket: each image is a sealed memfd passed with SCM_RIGHTS.
// No file under /tmp is involved.
//
// Request  = one packet: RequestHeader + 2 fds (old image, new image).
// Response = one packet: ResponseHeader + 1 fd (memfd holding a UTF-8 JSON payload of payloadBytes), or 0 fds when
//            payloadBytes == 0. Payload: {"generation", "status": "ok"|"refused", "reason", "leaves": [...same shape as
//            results/acceptance/<label>/<case>/selected-regions.json...], "splits": [...], "timings_ms": {...},
//            "prep_ms", "evidence": {...}}.
// The header repeats generation + status so a client can drop a stale or refused answer without reading the payload.
// Requests are served in order, one at a time; a client that already started a newer generation for the same window
// simply ignores older answers (the worker does not cancel).
#include "types.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace prep::ipc {

constexpr uint32_t REQUEST_MAGIC = 0x31515250;   // "PRQ1"
constexpr uint32_t RESPONSE_MAGIC = 0x31535250;  // "PRS1"
constexpr uint32_t VERSION = 1;

enum class Format : uint32_t {
    BGR8 = 1,   // 3 bytes per pixel, B G R (the analysis-image convention of the Python reference)
    BGRX8 = 2,  // 4 bytes per pixel, B G R X (DRM XRGB8888 / ARGB8888 little-endian); X ignored
};

struct ImageDesc {
    uint32_t width = 0, height = 0;  // analysis resolution (the FrameSource/Presenter downscales; 1/4 device px today)
    uint32_t stride = 0;             // bytes per row
    uint32_t offset = 0;             // byte offset of row 0 in the memfd
    Format format = Format::BGR8;
};

struct RequestHeader {
    uint32_t magic = REQUEST_MAGIC;
    uint32_t version = VERSION;
    uint64_t generation = 0;  // per-window resize generation (D1-7), echoed back
    uint64_t window = 0;      // window id (address), echoed back; the worker keeps no per-window state
    ImageDesc oldImage, newImage;
    uint32_t reserved[4] = {};
};

enum class Status : uint32_t { Ok = 0, Refused = 1 };

struct ResponseHeader {
    uint32_t magic = RESPONSE_MAGIC;
    uint32_t version = VERSION;
    uint64_t generation = 0;
    uint64_t window = 0;
    Status status = Status::Ok;
    uint32_t payloadBytes = 0;
    double prepMs = 0;        // worker-side preparation time (excludes socket + payload encoding)
    uint32_t reserved[4] = {};
};

static_assert(sizeof(RequestHeader) == 80 && sizeof(ResponseHeader) == 56, "wire layout is part of the contract");

std::string socketPath();

// Copies a BGR8 image into a sealed memfd; returns the fd (caller closes).
int imageMemfd(const cv::Mat& bgr, ImageDesc& desc);
// Maps an image memfd read-only and converts it to a CV_8UC3 BGR Mat (copy); throws on a malformed descriptor.
cv::Mat readImage(int fd, const ImageDesc& desc);

void sendPacket(int socket, const void* data, size_t bytes, const int* fds, int count);
// Returns bytes received (0 on orderly shutdown); fds[] receives up to maxFds descriptors.
size_t receivePacket(int socket, void* data, size_t bytes, int* fds, int maxFds, int& received);

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
