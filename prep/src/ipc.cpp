#include "ipc.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace prep::ipc {

int imageMemfd(const cv::Mat& bgr, ImageDesc& desc) {
    CV_Assert(bgr.type() == CV_8UC3);
    desc = {uint32_t(bgr.cols), uint32_t(bgr.rows), uint32_t(bgr.cols * 3), 0, Format::BGR8};
    cv::Mat packed = bgr.isContinuous() ? bgr : bgr.clone();
    return sealedMemfd("resize-prep-image", packed.data, size_t(desc.stride) * desc.height);
}

cv::Mat readImage(int fd, const ImageDesc& desc) {
    size_t pixel = desc.format == Format::BGR8 ? 3 : desc.format == Format::BGRX8 || desc.format == Format::RGBX8 ? 4 : 0;
    if (!pixel || desc.width < 2 || desc.height < 2 || desc.width > 8192 || desc.height > 8192 || desc.stride < desc.width * pixel)
        throw std::runtime_error("malformed image descriptor");
    struct stat st{};
    if (fstat(fd, &st) != 0) throw std::runtime_error("fstat image fd");
    size_t need = size_t(desc.offset) + size_t(desc.stride) * (desc.height - 1) + desc.width * pixel;
    if (size_t(st.st_size) < need) throw std::runtime_error("image fd smaller than its descriptor");
    void* map = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) throw std::runtime_error("mmap image fd");
    cv::Mat view(int(desc.height), int(desc.width), pixel == 3 ? CV_8UC3 : CV_8UC4, static_cast<uint8_t*>(map) + desc.offset, desc.stride);
    cv::Mat out;
    if (pixel == 3) out = view.clone();
    else cv::cvtColor(view, out, desc.format == Format::RGBX8 ? cv::COLOR_RGBA2BGR : cv::COLOR_BGRA2BGR);
    munmap(map, size_t(st.st_size));
    return out;
}

Client::Client(const std::string& path) {
    fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) throw std::runtime_error("socket");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path.size() >= sizeof address.sun_path) throw std::runtime_error("socket path too long");
    std::strcpy(address.sun_path, path.c_str());
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0)
        throw std::runtime_error("cannot connect to prep worker at " + path + ": " + std::strerror(errno));
}

Client::~Client() {
    if (fd >= 0) close(fd);
}

Response Client::prepare(uint64_t generation, const cv::Mat& oldImage, const cv::Mat& newImage, uint64_t window) {
    auto started = std::chrono::steady_clock::now();
    RequestHeader request;
    request.generation = generation;
    request.window = window;
    int fds[2] = {imageMemfd(oldImage, request.oldImage), imageMemfd(newImage, request.newImage)};
    sendPacket(fd, &request, sizeof request, fds, 2);
    close(fds[0]);
    close(fds[1]);
    ResponseHeader header;
    int payload = -1, received = 0;
    size_t n = receivePacket(fd, &header, sizeof header, &payload, 1, received);
    if (n != sizeof header || header.magic != RESPONSE_MAGIC || header.version != VERSION) throw std::runtime_error("malformed worker response");
    Response out;
    out.generation = header.generation;
    std::string text;
    if (header.payloadBytes) {
        if (!received) throw std::runtime_error("worker response without payload fd");
        text.resize(header.payloadBytes);
        if (pread(payload, text.data(), text.size(), 0) != ssize_t(text.size())) throw std::runtime_error("short payload read");
        close(payload);
    }
    if (header.status != Status::Ok) {
        out.refusal = text.empty() ? "refused" : json::parse(text)["reason"].str();
        return out;
    }
    json::Value doc = json::parse(text);
    out.leaves = leavesFromJson(doc["leaves"]);
    for (auto& [k, v] : doc["timings_ms"].obj()) out.timings.emplace_back(k, v.num());
    out.evidence = doc["evidence"];
    out.roundTripMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return out;
}

} // namespace prep::ipc
