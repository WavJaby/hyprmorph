#include "Thumbnail.hpp"

#include "Readback.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace {
    // 4 slots: one relayout resizes a few windows at once, each sampled at most once per commit.
    Readback::CPool                     pool{4, Thumbnail::WIDTH, Thumbnail::MAX_HEIGHT};
    std::map<uint64_t, Thumbnail::STag> tags; // by readback ticket, while in flight
}

int Thumbnail::heightFor(double aspect) {
    return std::clamp(static_cast<int>(std::lround(WIDTH * aspect)), 1, MAX_HEIGHT);
}

bool Thumbnail::submit(unsigned texture, float u0, float v0, float uw, float vh, int width, int height, const STag& tag) {
    const auto ticket = pool.submit(texture, u0, v0, uw, vh, std::clamp(width, 1, WIDTH), std::clamp(height, 1, MAX_HEIGHT));
    if (!ticket)
        return false;
    tags[ticket] = tag;
    return true;
}

void Thumbnail::poll(const std::function<void(SFrame&&)>& done) {
    pool.poll([&](uint64_t ticket, int width, int height, std::vector<uint8_t>&& rgba) {
        tags.erase(tags.begin(), tags.find(ticket)); // tickets complete in order: earlier ones still here lost their readback
        done(SFrame{.tag = tags.extract(ticket).mapped(), .width = width, .height = height, .rgba = std::move(rgba)});
    });
}

void Thumbnail::destroy() {
    pool.destroy();
    tags.clear();
}

const Readback::SCounters& Thumbnail::counters() {
    return pool.counters();
}

const char* Thumbnail::failure() {
    return pool.failure();
}
