#include "wire.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

namespace prep::ipc {

std::string socketPath() {
    const char* runtime = std::getenv("XDG_RUNTIME_DIR");
    if (!runtime || !*runtime) throw std::runtime_error("XDG_RUNTIME_DIR is not set; pass a socket path");
    return std::string(runtime) + "/hyprmorph/prep.sock";
}

int sealedMemfd(const char* name, const void* data, size_t bytes) {
    int fd = memfd_create(name, MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) throw std::runtime_error(std::string("memfd_create: ") + std::strerror(errno));
    for (size_t done = 0; done < bytes;) {
        ssize_t n = write(fd, static_cast<const char*>(data) + done, bytes - done);
        if (n <= 0) {
            close(fd);
            throw std::runtime_error(std::string("write memfd: ") + std::strerror(errno));
        }
        done += size_t(n);
    }
    fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL);
    return fd;
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

} // namespace prep::ipc
