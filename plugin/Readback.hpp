// Asynchronous downsampled readback of a texture region: a compute shader box-filters the region into a small RGBA8 image, read back through
// a pool of fenced PBOs so the compositor never waits on the GPU. 4x4 bilinear taps per output pixel -- the S3 sampler's filter
// (experiments/native_readiness_sampler.cpp); with an integer 4:1 ratio the taps sit on texel centres, i.e. an exact 4x4 box average.
// Users: Thumbnail (readiness frames) and the scheduler's analysis images for the prep worker (main.cpp).
// All calls need the compositor's EGL context current and run on the compositor thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Readback {
    struct SCounters {
        unsigned long submitted = 0, busy = 0, completed = 0, glErrors = 0;
    };

    class CPool {
      public:
        // slots = readbacks that can be in flight at once (one PBO each). minWidth x minHeight = storage allocated at first use; a larger
        // request grows it, which has to wait for the readbacks in flight -- a pool with a known maximum passes it here and never waits.
        CPool(size_t slots, int minWidth, int minHeight);

        // Makes room for width x height images ahead of a batch of submits: growing replaces the storage, which is only possible while no
        // readback is in flight -- so a batch with images of different sizes reserves its largest first. false -> try again next frame.
        bool reserve(int width, int height);
        // Downsamples uv rect (u0, v0)-(u0+uw, v0+vh) of `texture` (v = 0 at the top row of the content) to width x height.
        // Returns a ticket, or 0 when not submitted (no free slot, capacity growing while readbacks are in flight, or GL init failed: see
        // failure()); the caller retries on a later frame.
        uint64_t submit(unsigned texture, float u0, float v0, float uw, float vh, int width, int height);
        // Hands every finished readback to `done` (ticket, width, height, RGBA rows top first), oldest submission first; stops at the first
        // unfinished one, so tickets complete in submission order. A failed readback is counted and dropped.
        void poll(const std::function<void(uint64_t, int, int, std::vector<uint8_t>&&)>& done);
        void destroy();

        const SCounters& counters() const {
            return m_counts;
        }
        const char* failure() const { // empty while working
            return m_failure.c_str();
        }

      private:
        struct SSlot {
            unsigned pbo   = 0;
            void*    fence = nullptr; // GLsync
            uint64_t ticket = 0;
            int      width = 0, height = 0;
        };
        bool init();
        bool grow(int width, int height);

        std::vector<SSlot> m_slots;
        unsigned           m_program = 0, m_image = 0, m_framebuffer = 0, m_sampler = 0;
        int                m_uSource = -1, m_uExtent = -1, m_uCrop = -1;
        int                m_minW, m_minH;
        int                m_capacityW = 0, m_capacityH = 0; // target image and PBO size; grows on demand, never shrinks
        uint64_t           m_tickets = 0;
        bool               m_initTried = false;
        std::string        m_failure;
        SCounters          m_counts;
    };
}
