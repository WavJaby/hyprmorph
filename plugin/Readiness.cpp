#include "Readiness.hpp"

#include "../readiness/readiness.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <variant>

namespace {
    using namespace readiness;

    uint64_t nowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    struct SBegin {
        uint64_t    window, generation, resizeNs;
        std::string dumpDir;
    };
    struct SEnd {
        uint64_t window;
    };
    struct SCommit {
        uint64_t window, commit;
    };
    using Input = std::variant<SBegin, SEnd, SCommit, Thumbnail::SFrame>;

    struct SGeneration {
        uint64_t                           window = 0, generation = 0, resizeNs = 0, latestCommit = 0;
        std::unique_ptr<Detector>          detector;
        std::shared_ptr<Thumbnail::SFrame> latest;
        bool                               fresh = false;
        uint64_t                           lastObservationNs = 0;
        int                                frames = 0, observations = 0, selections = 0, vetoes = 0, revocations = 0;
        std::optional<double>              firstSelectionMs;
        bool                               readyAtEnd = false;
        std::vector<double>                analysis;
        std::filesystem::path              dumpDir;  // empty = no dump
        std::ofstream                      manifest;
        int                                dumped = 0;
        std::optional<Thumbnail::SFrame>   aliasRegion; // D5-1: old content's shared region; the detector is built at the first frame
    };

    Readiness::SConfig              config;
    std::thread                     worker;
    std::mutex                      mutex;
    std::condition_variable         wake;
    bool                            stopping = false;
    std::vector<Input>              inputs;
    std::vector<Readiness::SEvent>  events;
    std::map<uint64_t, SGeneration> generations; // worker-only

    void emit(Readiness::SEvent event) {
        std::lock_guard lock{mutex};
        events.push_back(std::move(event));
    }

    // Decisions also go into the dump manifest as comments (replay skips them): plugin/test/dump_parity.py compares them against the
    // replay without depending on hyprland.log, which is flushed lazily.
    void decide(SGeneration& g, Readiness::SEvent::EKind kind, double elapsedMs) {
        if (g.manifest.is_open())
            g.manifest << std::format("# decision {} {:.3f}\n", kind == Readiness::SEvent::SELECTED ? "select" : kind == Readiness::SEvent::VETOED ? "veto" : "revoke",
                                      elapsedMs);
        emit({.kind = kind, .window = g.window, .generation = g.generation, .elapsedMs = elapsedMs, .text = {}});
    }

    void close(SGeneration& g) {
        auto a = g.analysis;
        std::ranges::sort(a);
        emit({.kind      = Readiness::SEvent::SUMMARY,
              .window    = g.window,
              .generation = g.generation,
              .text      = std::format(R"({{"window": "0x{:x}", "generation": {}, "frames": {}, "observations": {}, "selections": {}, "vetoes": {}, )"
                                       R"("revocations": {}, "first_selection_ms": {}, "ready_at_end": {}, "analysis_ms_median": {:.3f}, "analysis_ms_max": {:.3f}}})",
                                       g.window, g.generation, g.frames, g.observations, g.selections, g.vetoes, g.revocations,
                                       g.firstSelectionMs ? std::format("{:.1f}", *g.firstSelectionMs) : "null", g.readyAtEnd, a.empty() ? 0. : a[a.size() / 2],
                                       a.empty() ? 0. : a.back())});
        if (g.manifest.is_open())
            g.manifest << "end\n";
    }

    // RGB PPM: readiness_replay loads .ppm as well as .png.
    void dumpFrame(SGeneration& g, const Thumbnail::SFrame& frame) {
        const auto name = std::format("{:x}-{}-{:03d}.ppm", g.window, g.generation, g.dumped);
        std::ofstream out(g.dumpDir / name, std::ios::binary);
        out << "P6\n" << frame.width << ' ' << frame.height << "\n255\n";
        for (size_t i = 0; i < frame.rgba.size(); i += 4)
            out.write(reinterpret_cast<const char*>(&frame.rgba[i]), 3);
        g.manifest << "frame " << (g.dumpDir / name).string() << '\n';
        ++g.dumped;
    }

    void dumpImage(const std::filesystem::path& path, int width, int height, const uint8_t* data, size_t stride, size_t channels) {
        std::ofstream out(path, std::ios::binary);
        out << "P6\n" << width << ' ' << height << "\n255\n";
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c)
                    out.put(static_cast<char>(data[y * stride + x * channels + (channels == 1 ? 0 : c)]));
    }

    // D5-1 crop alias in the new frame's thumbnail space, as aliases_from() builds it for recorded traces: old content pasted top-left on
    // black, mask = the shared region inset by aliasInset. Built at the first frame because only frames carry the new thumbnail size.
    std::vector<Alias> buildAliases(SGeneration& g, int width, int height) {
        if (!g.aliasRegion)
            return {};
        const auto&          region = *g.aliasRegion;
        std::vector<uint8_t> rgb(size_t(width) * height * 3, 0);
        const int            w = std::min(region.width, width), h = std::min(region.height, height), inset = region.tag.aliasInset;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                    rgb[(size_t(y) * width + x) * 3 + c] = region.rgba[(size_t(y) * region.width + x) * 4 + c];
        Alias alias;
        alias.color = lowPass(Image{width, height, 3, size_t(width) * 3, rgb.data()}, Policy{});
        alias.mask.assign(size_t(width) * height, 0);
        for (int y = inset; y < h - inset; ++y)
            for (int x = inset; x < w - inset; ++x)
                alias.mask[size_t(y) * width + x] = 255;
        if (g.manifest.is_open()) {
            const auto image = g.dumpDir / std::format("{:x}-{}-alias.ppm", g.window, g.generation);
            const auto mask  = g.dumpDir / std::format("{:x}-{}-alias-mask.ppm", g.window, g.generation);
            dumpImage(image, width, height, rgb.data(), size_t(width) * 3, 3);
            dumpImage(mask, width, height, alias.mask.data(), size_t(width), 1);
            g.manifest << "alias " << image.string() << ' ' << mask.string() << '\n';
        }
        return {std::move(alias)};
    }

    void apply(Input& input) {
        if (auto* b = std::get_if<SBegin>(&input)) {
            if (auto old = generations.find(b->window); old != generations.end()) {
                close(old->second);
                generations.erase(old);
            }
            auto& g      = generations[b->window];
            g.window     = b->window;
            g.generation = b->generation;
            g.resizeNs   = b->resizeNs;
            if (!b->dumpDir.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(b->dumpDir, ec);
                g.manifest.open(std::filesystem::path{b->dumpDir} / std::format("{:x}-{}.manifest", g.window, g.generation));
                if (ec || !g.manifest)
                    emit({.kind = Readiness::SEvent::ERROR, .window = g.window, .generation = g.generation, .text = "cannot write dump dir " + b->dumpDir});
                else {
                    g.dumpDir = b->dumpDir;
                    g.manifest << "trial " << std::format("{:x}-{}", g.window, g.generation) << '\n';
                }
            }
        } else if (auto* e = std::get_if<SEnd>(&input)) {
            if (auto g = generations.find(e->window); g != generations.end()) {
                close(g->second);
                generations.erase(g);
            }
        } else if (auto* c = std::get_if<SCommit>(&input)) {
            if (auto g = generations.find(c->window); g != generations.end())
                g->second.latestCommit = c->commit;
        } else {
            auto& frame = std::get<Thumbnail::SFrame>(input);
            auto  g     = generations.find(frame.tag.window);
            if (g == generations.end() || g->second.generation != frame.tag.generation)
                return;
            if (frame.tag.alias) {
                g->second.aliasRegion = std::move(frame);
                return;
            }
            if (!g->second.detector)
                g->second.detector = std::make_unique<Detector>(Policy{}, buildAliases(g->second, frame.width, frame.height));
            g->second.latest = std::make_shared<Thumbnail::SFrame>(std::move(frame));
            g->second.fresh  = true;
            ++g->second.frames;
            if (g->second.manifest.is_open())
                dumpFrame(g->second, *g->second.latest);
        }
    }

    void observe(SGeneration& g, uint64_t now) {
        const auto& f        = *g.latest;
        const bool  pending  = g.latestCommit != f.tag.commit;
        if (!g.fresh && (pending || now - g.lastObservationNs < config.reobserveMs * 1e6))
            return;
        const double elapsedMs = (now - g.resizeNs) / 1e6;
        const Image  image{f.width, f.height, 4, size_t(f.width) * 4, f.rgba.data()};
        const auto   o = g.detector->observe(image, elapsedMs, Evidence{true, true, std::nullopt});
        g.fresh             = false;
        g.lastObservationNs = now;
        ++g.observations;
        g.analysis.push_back(o.analysisMs);
        if (g.manifest.is_open())
            g.manifest << std::format("observe {} {:.3f} 1 -1 1\n", g.dumped - 1, elapsedMs);
        if (o.revoked) {
            ++g.revocations;
            decide(g, Readiness::SEvent::REVOKED, elapsedMs);
        }
        bool vetoed = false;
        if (o.newlyReady) {
            const double sampleAgeMs = (now - f.tag.sampleNs) / 1e6;
            if (pending && sampleAgeMs > g.detector->policy().maximumSampleGapMs) {
                g.detector->cancelSelection();
                vetoed = true;
                ++g.vetoes;
                if (g.manifest.is_open())
                    g.manifest << "veto\n";
                decide(g, Readiness::SEvent::VETOED, elapsedMs);
            } else {
                ++g.selections;
                if (!g.firstSelectionMs)
                    g.firstSelectionMs = elapsedMs;
                decide(g, Readiness::SEvent::SELECTED, elapsedMs);
            }
        }
        g.readyAtEnd = o.ready && !vetoed;
    }

    void run() {
        std::unique_lock lock{mutex};
        while (!stopping) {
            // The re-observation tick is only for open generations; with none the thread sleeps until the next input.
            const auto pending = [] { return stopping || !inputs.empty(); };
            if (generations.empty())
                wake.wait(lock, pending);
            else
                wake.wait_for(lock, std::chrono::duration<double, std::milli>(config.reobserveMs), pending);
            auto batch = std::move(inputs);
            inputs.clear();
            lock.unlock();
            for (auto& input : batch)
                apply(input);
            const auto now = nowNs();
            for (auto it = generations.begin(); it != generations.end();) {
                auto& g = it->second;
                if ((now - g.resizeNs) / 1e6 > config.observeMs) {
                    close(g);
                    it = generations.erase(it);
                    continue;
                }
                if (g.latest)
                    observe(g, now);
                ++it;
            }
            lock.lock();
        }
        lock.unlock();
        for (auto& [_, g] : generations)
            close(g);
        generations.clear();
    }

    void push(Input&& input) {
        {
            std::lock_guard lock{mutex};
            if (!worker.joinable())
                return;
            inputs.push_back(std::move(input));
        }
        wake.notify_one();
    }
}

void Readiness::start(const SConfig& c) {
    if (worker.joinable())
        return;
    config   = c;
    stopping = false;
    worker   = std::thread{run};
}

void Readiness::stop() {
    {
        std::lock_guard lock{mutex};
        stopping = true;
    }
    wake.notify_one();
    if (worker.joinable())
        worker.join();
    inputs.clear();
}

void Readiness::beginGeneration(uint64_t window, uint64_t generation, uint64_t resizeNs, std::string dumpDir) {
    push(SBegin{window, generation, resizeNs, std::move(dumpDir)});
}

void Readiness::endGeneration(uint64_t window) {
    push(SEnd{window});
}

void Readiness::noteCommit(uint64_t window, uint64_t commit) {
    push(SCommit{window, commit});
}

void Readiness::submit(Thumbnail::SFrame&& frame) {
    push(std::move(frame));
}

std::vector<Readiness::SEvent> Readiness::drain() {
    std::lock_guard lock{mutex};
    return std::exchange(events, {});
}
