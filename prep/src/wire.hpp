#pragma once
// Prep worker wire format (preparation runs out of process, keyed by resize generation). Standard library + POSIX only: the
// compositor plugin compiles this file and wire.cpp without OpenCV; ipc.hpp adds the cv::Mat conveniences for the worker and the tools.
//
// Transport: one AF_UNIX SOCK_SEQPACKET connection per client -- a socketpair end inherited from the compositor (prep-worker --fd), or a
// listening socket at $XDG_RUNTIME_DIR/resize-transition/prep.sock for the tools (socketPath()). Pixels never travel in the socket: each
// image is a sealed memfd passed with SCM_RIGHTS. No file under /tmp is involved.
//
// Request  = one packet: RequestHeader + 2 fds (old image, new image).
// Response = one packet: ResponseHeader + 1 fd (memfd holding a UTF-8 JSON payload of payloadBytes), or 0 fds when payloadBytes == 0.
//            Payload: {"generation", "status": "ok"|"refused", "reason", "leaves": [...prep::Leaf JSON
//            (types.hpp)...], "splits": [...], "timings_ms": {...}, "prep_ms", "evidence": {...}}.
// The header repeats window + generation + candidate + status so a client can drop a stale or refused answer without reading the payload.
// Requests are served in order, one at a time; a client that already started a newer request for the same window simply ignores older
// answers (the worker does not cancel).
#include <cstddef>
#include <cstdint>
#include <string>

namespace prep::ipc {

constexpr uint32_t REQUEST_MAGIC = 0x31515250;   // "PRQ1"
constexpr uint32_t RESPONSE_MAGIC = 0x31535250;  // "PRS1"
constexpr uint32_t VERSION = 1;

enum class Format : uint32_t {
    BGR8 = 1,   // 3 bytes per pixel, B G R (OpenCV's analysis-image convention)
    BGRX8 = 2,  // 4 bytes per pixel, B G R X (DRM XRGB8888 / ARGB8888 little-endian); X ignored
    RGBX8 = 3,  // 4 bytes per pixel, R G B X (a GL_RGBA readback); X ignored
};

struct ImageDesc {
    uint32_t width = 0, height = 0;  // analysis resolution (the plugin downscales; 1/4 device px today)
    uint32_t stride = 0;             // bytes per row
    uint32_t offset = 0;             // byte offset of row 0 in the memfd
    Format format = Format::BGR8;
};

struct RequestHeader {
    uint32_t magic = REQUEST_MAGIC;
    uint32_t version = VERSION;
    uint64_t generation = 0;  // per-window resize generation, echoed back
    uint64_t window = 0;      // window id (address), echoed back; the worker keeps no per-window state
    ImageDesc oldImage, newImage;
    uint32_t candidate = 0;   // readiness candidate within the generation (each restart of the confirm window is a new one), echoed back
    uint32_t reserved[3] = {};
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
    uint32_t candidate = 0;
    uint32_t reserved[3] = {};
};

static_assert(sizeof(RequestHeader) == 80 && sizeof(ResponseHeader) == 56, "wire layout is part of the contract");

std::string socketPath();

// Copies `bytes` into a new memfd and seals it (shrink, grow, write): the receiver may map it without trusting the sender to stop writing
// or to keep the size. Returns the fd (caller closes); throws std::runtime_error.
int sealedMemfd(const char* name, const void* data, size_t bytes);

void sendPacket(int socket, const void* data, size_t bytes, const int* fds, int count);
// Returns bytes received (0 on orderly shutdown); fds[] receives up to maxFds descriptors.
size_t receivePacket(int socket, void* data, size_t bytes, int* fds, int maxFds, int& received);

} // namespace prep::ipc
