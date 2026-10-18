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

std::string socketPath() {
    const char* runtime = std::getenv("XDG_RUNTIME_DIR");
    if (!runtime || !*runtime) throw std::runtime_error("XDG_RUNTIME_DIR is not set; pass a socket path");
    return std::string(runtime) + "/resize-transition/prep.sock";
}

int imageMemfd(const cv::Mat& bgr, ImageDesc& desc) {
    CV_Assert(bgr.type() == CV_8UC3);
    desc = {uint32_t(bgr.cols), uint32_t(bgr.rows), uint32_t(bgr.cols * 3), 0, Format::BGR8};
    size_t bytes = size_t(desc.stride) * desc.height;
    int fd = memfd_create("resize-prep-image", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) throw std::runtime_error(std::string("memfd_create: ") + std::strerror(errno));
    if (ftruncate(fd, off_t(bytes)) != 0) throw std::runtime_error("ftruncate image memfd");
    void* map = mmap(nullptr, bytes, PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) throw std::runtime_error("mmap image memfd");
    for (int y = 0; y < bgr.rows; ++y) std::memcpy(static_cast<uint8_t*>(map) + size_t(y) * desc.stride, bgr.ptr(y), desc.stride);
    munmap(map, bytes);
    // Sealed: the worker may map it without trusting the sender to stop writing or to keep the size.
    fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL);
    return fd;
}

cv::Mat readImage(int fd, const ImageDesc& desc) {
    size_t pixel = desc.format == Format::BGR8 ? 3 : desc.format == Format::BGRX8 ? 4 : 0;
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
    else cv::cvtColor(view, out, cv::COLOR_BGRA2BGR);
    munmap(map, size_t(st.st_size));
    return out;
}

void sendPacket(int socket, const void* data, size_t bytes, const int* fds, int count) {
    iovec iov{const_cast<void*>(data), bytes};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * 4)] = {};
    if (count) {
        msg.msg_control = control;
        msg.msg_controllen = CMSG_SPACE(sizeof(int) * size_t(count));
        cmsghdr* c = CMSG_FIRSTHDR(&msg);
        c->cmsg_level = SOL_SOCKET;
        c->cmsg_type = SCM_RIGHTS;
        c->cmsg_len = CMSG_LEN(sizeof(int) * size_t(count));
        std::memcpy(CMSG_DATA(c), fds, sizeof(int) * size_t(count));
    }
    if (sendmsg(socket, &msg, MSG_NOSIGNAL) != ssize_t(bytes)) throw std::runtime_error(std::string("sendmsg: ") + std::strerror(errno));
}

size_t receivePacket(int socket, void* data, size_t bytes, int* fds, int maxFds, int& received) {
    iovec iov{data, bytes};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * 4)] = {};
    msg.msg_control = control;
    msg.msg_controllen = sizeof control;
    ssize_t n = recvmsg(socket, &msg, MSG_CMSG_CLOEXEC);
    if (n < 0) throw std::runtime_error(std::string("recvmsg: ") + std::strerror(errno));
    received = 0;
    for (cmsghdr* c = CMSG_FIRSTHDR(&msg); c; c = CMSG_NXTHDR(&msg, c))
        if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS) {
            int count = int((c->cmsg_len - CMSG_LEN(0)) / sizeof(int));
            for (int k = 0; k < count; ++k) {
                int fd;
                std::memcpy(&fd, CMSG_DATA(c) + sizeof(int) * size_t(k), sizeof(int));
                if (received < maxFds) fds[received++] = fd;
                else close(fd);
            }
        }
    if (msg.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) throw std::runtime_error("truncated packet");
    return size_t(n);
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
