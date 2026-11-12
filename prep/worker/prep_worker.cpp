// prep-worker: long-lived mesh preparation service (D1-1: out of the compositor process).
//
//   prep-worker [--socket PATH | --fd N] [--node /dev/dri/renderD128]
//
// Owns one headless EGL/GLES context on the render node and the compiled NCC/CG kernels for its whole life, so a request pays no context,
// shader or JIT cost after the first one. Serves the contract in src/wire.hpp, requests in order.
// --socket (default $XDG_RUNTIME_DIR/resize-transition/prep.sock, directory 0700): listens, one client connection at a time, runs until
// killed -- for the tools. --fd N: serves the already-connected SOCK_SEQPACKET socket inherited as descriptor N and exits when the peer
// closes it -- how the compositor plugin runs it, so the worker's life is the connection's (no path, no orphan after a compositor crash).
// Default: mutating (creates the socket; runs until killed).
#include "../src/ipc.hpp"
#include "../src/pipeline.hpp"

#include <opencv2/core/utility.hpp>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace prep;

namespace {

std::string socketFile;
void cleanup(int) {
    if (!socketFile.empty()) unlink(socketFile.c_str());
    _exit(0);
}

void respond(int client, const ipc::RequestHeader& request, ipc::Status status, double prepMs, const json::Value& payload) {
    ipc::ResponseHeader header;
    header.generation = request.generation;
    header.window = request.window;
    header.selection = request.selection;
    header.status = status;
    header.prepMs = prepMs;
    std::string text = json::dump(payload);
    header.payloadBytes = uint32_t(text.size());
    int fd = ipc::sealedMemfd("resize-prep-payload", text.data(), text.size());
    ipc::sendPacket(client, &header, sizeof header, &fd, 1);
    close(fd);
}

void serve(int client, Preparer& preparer) {
    Config config;
    while (true) {
        ipc::RequestHeader request;
        int fds[2] = {-1, -1}, received = 0;
        size_t n = ipc::receivePacket(client, &request, sizeof request, fds, 2, received);
        if (n == 0) return;
        json::Value payload = json::Value::object();
        payload.set("generation", double(request.generation));
        try {
            if (n != sizeof request || request.magic != ipc::REQUEST_MAGIC || request.version != ipc::VERSION || received != 2)
                throw std::invalid_argument("malformed request (header or fd count)");
            cv::Mat oldImage = ipc::readImage(fds[0], request.oldImage), newImage = ipc::readImage(fds[1], request.newImage);
            for (int fd : fds) close(fd);
            Result result = preparer.run(oldImage, newImage, config);
            json::Value timings = json::Value::object();
            for (auto& [k, v] : result.timings) timings.set(k, v);
            payload.set("status", "ok");
            payload.set("leaves", toJson(result.leaves));
            json::Array splits;
            for (auto& s : result.splits) splits.push_back(s.toJson());
            payload.set("splits", splits);
            payload.set("timings_ms", timings);
            payload.set("prep_ms", result.prepMs());
            payload.set("evidence", result.evidence);
            respond(client, request, ipc::Status::Ok, result.prepMs(), payload);
        } catch (const std::exception& e) {
            for (int k = 0; k < received; ++k) close(fds[k]);
            // A refusal is an answer: the client falls back to the degenerate (plain-scale) mesh, D1-4.
            payload.set("status", "refused");
            payload.set("reason", e.what());
            respond(client, request, ipc::Status::Refused, 0, payload);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string node = "/dev/dri/renderD128";
        int inherited = -1;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--socket" && i + 1 < argc) socketFile = argv[++i];
            else if (a == "--fd" && i + 1 < argc) inherited = std::atoi(argv[++i]);
            else if (a == "--node" && i + 1 < argc) node = argv[++i];
            else {
                std::fprintf(stderr, "usage: prep-worker [--socket PATH | --fd N] [--node RENDER_NODE]\n");
                return 2;
            }
        }
        if (inherited >= 0 && !socketFile.empty()) throw std::invalid_argument("--socket and --fd exclude each other");
        cv::setNumThreads(1);
        auto started = std::chrono::steady_clock::now();
        gles::Context context(node);
        Preparer preparer(context);
        double startup = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        double warm = preparer.warm();
        if (inherited >= 0) {
            // Requests sent during start-up waited in the socket buffer.
            std::fprintf(stderr, "prep-worker: serving fd %d (context + kernels %.1f ms, warm-up %.1f ms)\n", inherited, startup, warm);
            serve(inherited, preparer);
            return 0;
        }
        if (socketFile.empty()) socketFile = ipc::socketPath();
        std::filesystem::path folder = std::filesystem::path(socketFile).parent_path();
        std::filesystem::create_directories(folder);
        std::filesystem::permissions(folder, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
        int listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::strncpy(address.sun_path, socketFile.c_str(), sizeof address.sun_path - 1);
        unlink(socketFile.c_str());
        if (bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0 || listen(listener, 4) != 0)
            throw std::runtime_error("cannot listen on " + socketFile + ": " + std::strerror(errno));
        std::signal(SIGTERM, cleanup);
        std::signal(SIGINT, cleanup);
        std::fprintf(stderr, "prep-worker: listening on %s (context + kernels %.1f ms, warm-up %.1f ms)\n", socketFile.c_str(), startup, warm);
        while (true) {
            int client = accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
            if (client < 0) continue;
            try {
                serve(client, preparer);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "prep-worker: connection dropped: %s\n", e.what());
            }
            close(client);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prep-worker: %s\n", e.what());
        if (!socketFile.empty()) unlink(socketFile.c_str());
        return 1;
    }
}
