// Client of the out-of-process mesh preparation worker. The plugin spawns prep-worker as
// a child holding one end of a socketpair and talks prep/src/wire.hpp over it from a client thread, so the compositor thread never blocks
// on the worker. The worker's life is the connection's: it exits when the plugin closes its end (unload, or a compositor crash).
// No worker, a dead worker or a refusal all mean the same to the caller: no mesh, the Presenter keeps the plain-scale degenerate mesh.
// A dead worker is not restarted; it comes back with the next start() (plugin reload, or a changed worker path).
#pragma once

#include "../prep/src/types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Prep {
    // Analysis image as Readback delivers it: RGBA, top row first.
    struct SImage {
        int                  width = 0, height = 0;
        std::vector<uint8_t> rgba;
    };
    // (window, generation, candidate) identify the readiness candidate the images belong to and come back with the answer.
    struct SRequest {
        uint64_t window = 0, generation = 0;
        uint32_t candidate = 0;
        SImage   oldImage, newImage;
    };
    struct SAnswer {
        uint64_t    window = 0, generation = 0;
        uint32_t    candidate = 0;
        bool        ok = false;
        std::string refusal;         // worker's reason when !ok
        std::vector<prep::Leaf> leaves; // the mesh: regions in analysis px of the two request images
        int         oldWidth = 0, oldHeight = 0, newWidth = 0, newHeight = 0; // those images' sizes
        double      prepMs = 0;      // worker-side preparation
        double      roundTripMs = 0; // request() -> answer parsed, incl. the wait behind earlier requests
    };

    // workerPath value that runs the worker image linked into this plugin (EmbeddedWorker) instead of a file.
    inline const std::string EMBEDDED = "embedded";

    // Spawns `workerPath --fd N --node renderNode` and starts the client thread; stops a previous worker first. false -> reason in `error`, no worker.
    bool start(const std::string& workerPath, const std::string& renderNode, std::string& error);
    // Closes the connection, joins the thread (it executes this library's code: must run before unload) and reaps the worker.
    void stop();
    // A worker is connected: a request made now will be answered (or reported by takeFailure()).
    bool running();
    // Path of the running or last started worker; empty before the first start().
    const std::string& workerPath();
    // Render node handed to that worker.
    const std::string& renderNode();

    // Queues a request. An unsent request of the same window is replaced; dropped when no worker is running.
    void request(SRequest&& request);
    std::vector<SAnswer> drain();
    // Reason the worker connection went down, once per failure.
    std::optional<std::string> takeFailure();

    // dropped = requests made with no worker; replaced = requests superseded before they were sent.
    struct SCounters {
        unsigned long requests = 0, dropped = 0, replaced = 0, ok = 0, refused = 0;
    };
    SCounters counters();
}
