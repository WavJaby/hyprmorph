#include "Prep.hpp"
#include "EmbeddedWorker.hpp"

#include "../prep/src/json.hpp"
#include "../prep/src/wire.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <deque>
#include <mutex>
#include <format>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;

namespace {
    using Clock = std::chrono::steady_clock;
    namespace wire = prep::ipc;

    // Descriptor number the worker's end of the socketpair gets in the child.
    constexpr int WORKER_FD = 3;

    // Spawns `executable` with the worker's argv; posix_spawn returns once the child has exec'd (or failed to).
    int spawnWorker(const std::string& executable, char* const argv[], pid_t& child, int workerEnd) {
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        // dup2 clears CLOEXEC on the worker's end; closefrom keeps every other compositor descriptor (DRM, clients) out of the worker.
        posix_spawn_file_actions_adddup2(&actions, workerEnd, WORKER_FD);
        posix_spawn_file_actions_addclosefrom_np(&actions, WORKER_FD + 1);
        // The compositor blocks and ignores signals for its own event loop; inherited, they would leave the worker deaf to SIGTERM.
        posix_spawnattr_t attributes;
        posix_spawnattr_init(&attributes);
        sigset_t none, all;
        sigemptyset(&none);
        sigfillset(&all);
        posix_spawnattr_setsigmask(&attributes, &none);
        posix_spawnattr_setsigdefault(&attributes, &all);
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
        const int spawned = posix_spawn(&child, executable.c_str(), &actions, &attributes, argv, environ);
        posix_spawnattr_destroy(&attributes);
        posix_spawn_file_actions_destroy(&actions);
        return spawned;
    }

    struct SQueued {
        Prep::SRequest    request;
        Clock::time_point queuedAt;
    };

    std::string                path;
    std::string                node;
    pid_t                      worker = -1;
    int                        connection = -1;
    std::thread                client;
    std::mutex                 mutex;
    std::condition_variable    wake;
    bool                       stopping = false, up = false;
    std::deque<SQueued>        queue;
    std::vector<Prep::SAnswer> answers;
    std::optional<std::string> failure;
    Prep::SCounters            counts;

    wire::ImageDesc describe(const Prep::SImage& image) {
        return {uint32_t(image.width), uint32_t(image.height), uint32_t(image.width) * 4, 0, wire::Format::RGBX8};
    }

    // One blocking exchange; throws when the connection is unusable.
    Prep::SAnswer exchange(const SQueued& q) {
        wire::RequestHeader header;
        header.generation = q.request.generation;
        header.window     = q.request.window;
        header.candidate  = q.request.candidate;
        header.oldImage   = describe(q.request.oldImage);
        header.newImage   = describe(q.request.newImage);
        int fds[2]        = {-1, -1};
        try {
            fds[0] = wire::sealedMemfd("resize-prep-old", q.request.oldImage.rgba.data(), q.request.oldImage.rgba.size());
            fds[1] = wire::sealedMemfd("resize-prep-new", q.request.newImage.rgba.data(), q.request.newImage.rgba.size());
            wire::sendPacket(connection, &header, sizeof header, fds, 2);
        } catch (...) {
            for (int fd : fds)
                if (fd >= 0)
                    close(fd);
            throw;
        }
        close(fds[0]);
        close(fds[1]);

        wire::ResponseHeader response;
        int                  payload = -1, received = 0;
        const size_t         n = wire::receivePacket(connection, &response, sizeof response, &payload, 1, received);
        std::string          text;
        if (received) {
            text.resize(response.payloadBytes);
            const bool complete = pread(payload, text.data(), text.size(), 0) == ssize_t(text.size());
            close(payload);
            if (!complete)
                throw std::runtime_error("short payload read");
        }
        if (n == 0)
            throw std::runtime_error("worker closed the connection");
        if (n != sizeof response || response.magic != wire::RESPONSE_MAGIC || response.version != wire::VERSION)
            throw std::runtime_error("malformed worker response");

        Prep::SAnswer answer;
        answer.window     = response.window;
        answer.generation = response.generation;
        answer.candidate  = response.candidate;
        answer.ok         = response.status == wire::Status::Ok;
        answer.prepMs = response.prepMs;
        if (!text.empty()) {
            const auto doc = prep::json::parse(text);
            if (answer.ok) {
                if (const auto* leaves = doc.find("leaves"); leaves && leaves->isArray())
                    answer.leaves = prep::leavesFromJson(*leaves);
            } else if (const auto* reason = doc.find("reason"); reason && reason->isString())
                answer.refusal = reason->str();
        }
        answer.oldWidth    = q.request.oldImage.width;
        answer.oldHeight   = q.request.oldImage.height;
        answer.newWidth    = q.request.newImage.width;
        answer.newHeight   = q.request.newImage.height;
        answer.roundTripMs = std::chrono::duration<double, std::milli>(Clock::now() - q.queuedAt).count();
        return answer;
    }

    void run() {
        std::unique_lock lock{mutex};
        while (true) {
            wake.wait(lock, [] { return stopping || !queue.empty(); });
            if (stopping)
                return;
            const SQueued q = std::move(queue.front());
            queue.pop_front();
            lock.unlock();
            std::optional<Prep::SAnswer> answer;
            std::string                  error;
            try {
                answer = exchange(q);
            } catch (const std::exception& e) { error = e.what(); }
            lock.lock();
            if (!answer) {
                // stop() shuts the socket down to wake a blocked exchange: that is not a failure.
                if (!stopping)
                    failure = error;
                up = false;
                queue.clear();
                return;
            }
            ++(answer->ok ? counts.ok : counts.refused);
            answers.push_back(std::move(*answer));
        }
    }
}

bool Prep::start(const std::string& workerPath, const std::string& renderNode, std::string& error) {
    stop();
    path = workerPath;
    node = renderNode;
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, pair) != 0) {
        error = std::string{"socketpair: "} + std::strerror(errno);
        return false;
    }
    const bool        embedded = workerPath == EMBEDDED;
    const std::string fd       = std::to_string(WORKER_FD);
    char* const       argv[]   = {const_cast<char*>(embedded ? "prep-worker" : workerPath.c_str()), const_cast<char*>("--fd"), const_cast<char*>(fd.c_str()),
                                  const_cast<char*>("--node"), const_cast<char*>(renderNode.c_str()), nullptr};
    int               spawned  = 0;
    if (embedded) {
        // The worker image linked into this .so, run from a sealed memfd: an install (hyprpm keeps only the .so) needs no second file.
        // Exec'd by the parent's /proc path -- the child's closefrom has already closed its own copy of the memfd by then.
        int image = -1;
        try {
            image = prep::ipc::sealedMemfd("prep-worker", EmbeddedWorker::begin(), EmbeddedWorker::size());
        } catch (const std::exception& e) {
            close(pair[0]);
            close(pair[1]);
            error = e.what();
            return false;
        }
        spawned = spawnWorker(std::format("/proc/{}/fd/{}", getpid(), image), argv, worker, pair[1]);
        close(image);
    } else
        spawned = spawnWorker(workerPath, argv, worker, pair[1]);
    close(pair[1]);
    if (spawned != 0) {
        close(pair[0]);
        worker = -1;
        error  = "cannot spawn " + workerPath + ": " + std::strerror(spawned);
        return false;
    }
    connection = pair[0];
    stopping   = false;
    up         = true;
    client     = std::thread{run};
    return true;
}

void Prep::stop() {
    {
        std::lock_guard lock{mutex};
        stopping = true;
    }
    wake.notify_one();
    if (connection >= 0)
        shutdown(connection, SHUT_RDWR);
    if (client.joinable())
        client.join();
    if (connection >= 0)
        close(connection);
    connection = -1;
    if (worker > 0) {
        // SIGTERM as well as the closed connection: during its start-up warm-up (up to ~2 s) the worker is not reading the socket yet,
        // and unload must not wait that long on the compositor thread.
        kill(worker, SIGTERM);
        waitpid(worker, nullptr, 0);
    }
    worker = -1;
    up     = false;
    queue.clear();
    answers.clear();
    failure.reset();
}

bool Prep::running() {
    std::lock_guard lock{mutex};
    return up;
}

const std::string& Prep::workerPath() {
    return path;
}

const std::string& Prep::renderNode() {
    return node;
}

void Prep::request(SRequest&& request) {
    {
        std::lock_guard lock{mutex};
        ++counts.requests;
        if (!up) {
            ++counts.dropped;
            return;
        }
        counts.replaced += std::erase_if(queue, [&](const SQueued& q) { return q.request.window == request.window; });
        queue.push_back({std::move(request), Clock::now()});
    }
    wake.notify_one();
}

std::vector<Prep::SAnswer> Prep::drain() {
    std::lock_guard lock{mutex};
    return std::exchange(answers, {});
}

std::optional<std::string> Prep::takeFailure() {
    std::lock_guard lock{mutex};
    return std::exchange(failure, std::nullopt);
}

Prep::SCounters Prep::counters() {
    std::lock_guard lock{mutex};
    return counts;
}
