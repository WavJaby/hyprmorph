#include "Thumbnail.hpp"

#include <GLES3/gl31.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace {
    // 4 slots: one relayout resizes a few windows at once, each sampled at most once per commit.
    constexpr size_t POOL  = 4;
    constexpr size_t BYTES = size_t(Thumbnail::WIDTH) * Thumbnail::MAX_HEIGHT * 4;

    struct SSlot {
        GLuint          pbo   = 0;
        GLsync          fence = nullptr;
        uint64_t        order = 0;
        int             width = 0, height = 0;
        Thumbnail::STag tag;
    };

    std::array<SSlot, POOL> slots;
    GLuint                  program = 0, image = 0, framebuffer = 0, sampler = 0;
    GLint                   uSource = -1, uExtent = -1, uCrop = -1;
    uint64_t                submissions = 0;
    bool                    initTried = false;
    std::string             failureText;
    Thumbnail::SCounters    counts;

    // Hyprland caches GL state; everything touched here is put back.
    struct SSavedState {
        GLint program, activeTexture, texture, readFramebuffer, packBuffer, packAlignment, rowLength, skipRows, skipPixels, boundSampler;
        GLint image, imageLevel, imageLayered, imageLayer, imageAccess, imageFormat;
        SSavedState() {
            glGetIntegerv(GL_CURRENT_PROGRAM, &program);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
            glActiveTexture(GL_TEXTURE0);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &texture);
            glGetIntegerv(GL_SAMPLER_BINDING, &boundSampler);
            glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &readFramebuffer);
            glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &packBuffer);
            glGetIntegerv(GL_PACK_ALIGNMENT, &packAlignment);
            glGetIntegerv(GL_PACK_ROW_LENGTH, &rowLength);
            glGetIntegerv(GL_PACK_SKIP_ROWS, &skipRows);
            glGetIntegerv(GL_PACK_SKIP_PIXELS, &skipPixels);
            glGetIntegeri_v(GL_IMAGE_BINDING_NAME, 0, &image);
            glGetIntegeri_v(GL_IMAGE_BINDING_LEVEL, 0, &imageLevel);
            glGetIntegeri_v(GL_IMAGE_BINDING_LAYERED, 0, &imageLayered);
            glGetIntegeri_v(GL_IMAGE_BINDING_LAYER, 0, &imageLayer);
            glGetIntegeri_v(GL_IMAGE_BINDING_ACCESS, 0, &imageAccess);
            glGetIntegeri_v(GL_IMAGE_BINDING_FORMAT, 0, &imageFormat);
        }
        ~SSavedState() {
            glBindImageTexture(0, image, imageLevel, imageLayered, imageLayer, imageAccess, imageFormat);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, readFramebuffer);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, packBuffer);
            glPixelStorei(GL_PACK_ALIGNMENT, packAlignment);
            glPixelStorei(GL_PACK_ROW_LENGTH, rowLength);
            glPixelStorei(GL_PACK_SKIP_ROWS, skipRows);
            glPixelStorei(GL_PACK_SKIP_PIXELS, skipPixels);
            glBindSampler(0, boundSampler);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(activeTexture);
            glUseProgram(program);
        }
    };

    bool init() {
        initTried                    = true;
        constexpr const char* SOURCE = R"(#version 310 es
            precision highp float;
            layout(local_size_x=8, local_size_y=8) in;
            layout(rgba8, binding=0) writeonly uniform highp image2D resultImage;
            uniform sampler2D sourceImage;
            uniform ivec2 extent;
            uniform vec4 crop;
            void main() {
                ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
                if (any(greaterThanEqual(pixel, extent))) return;
                vec4 sum = vec4(0.0);
                for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x) {
                    vec2 uv = (vec2(pixel) + (vec2(x, y) + 0.5) / 4.0) / vec2(extent);
                    sum += textureLod(sourceImage, crop.xy + uv * crop.zw, 0.0);
                }
                imageStore(resultImage, pixel, sum / 16.0);
            })";
        const GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(shader, 1, &SOURCE, nullptr);
        glCompileShader(shader);
        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            std::array<char, 1024> log{};
            glGetShaderInfoLog(shader, log.size(), nullptr, log.data());
            failureText = std::string{"compute shader: "} + log.data();
            glDeleteShader(shader);
            return false;
        }
        program = glCreateProgram();
        glAttachShader(program, shader);
        glLinkProgram(program);
        glDeleteShader(shader);
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            failureText = "compute program link failed";
            return false;
        }
        uSource = glGetUniformLocation(program, "sourceImage");
        uExtent = glGetUniformLocation(program, "extent");
        uCrop   = glGetUniformLocation(program, "crop");

        glGenTextures(1, &image);
        glBindTexture(GL_TEXTURE_2D, image);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, Thumbnail::WIDTH, Thumbnail::MAX_HEIGHT);
        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, image, 0);
        if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            failureText = "thumbnail framebuffer incomplete";
            return false;
        }
        glGenSamplers(1, &sampler);
        glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        for (auto& slot : slots) {
            glGenBuffers(1, &slot.pbo);
            glBindBuffer(GL_PIXEL_PACK_BUFFER, slot.pbo);
            glBufferData(GL_PIXEL_PACK_BUFFER, BYTES, nullptr, GL_STREAM_READ);
        }
        return true;
    }
}

int Thumbnail::heightFor(double aspect) {
    return std::clamp(static_cast<int>(std::lround(WIDTH * aspect)), 1, MAX_HEIGHT);
}

bool Thumbnail::submit(unsigned texture, float u0, float v0, float uw, float vh, int width, int height, const STag& tag) {
    if (!failureText.empty())
        return false;
    const auto free = std::ranges::find_if(slots, [](const SSlot& s) { return !s.fence; });
    if (free == slots.end()) {
        ++counts.busy;
        return false;
    }
    SSavedState saved;
    if (!initTried && !init())
        return false;

    width  = std::clamp(width, 1, WIDTH);
    height = std::clamp(height, 1, MAX_HEIGHT);
    glUseProgram(program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindSampler(0, sampler);
    glBindImageTexture(0, image, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glUniform1i(uSource, 0);
    glUniform2i(uExtent, width, height);
    glUniform4f(uCrop, u0, v0, uw, vh);
    glDispatchCompute((width + 7) / 8, (height + 7) / 8, 1);
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, free->pbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    free->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
    if (glGetError() != GL_NO_ERROR)
        ++counts.glErrors;
    free->order  = ++submissions;
    free->width  = width;
    free->height = height;
    free->tag    = tag;
    ++counts.submitted;
    return true;
}

void Thumbnail::poll(const std::function<void(SFrame&&)>& done) {
    std::array<SSlot*, POOL> ordered;
    for (size_t i = 0; i < POOL; ++i)
        ordered[i] = &slots[i];
    std::ranges::sort(ordered, {}, &SSlot::order);
    if (std::ranges::none_of(ordered, [](const SSlot* s) { return s->fence; }))
        return;

    SSavedState saved;
    for (auto* slot : ordered) {
        if (!slot->fence)
            continue;
        const auto status = glClientWaitSync(slot->fence, 0, 0);
        if (status == GL_TIMEOUT_EXPIRED)
            break;
        SFrame frame{.tag = slot->tag, .width = slot->width, .height = slot->height, .rgba = {}};
        if (status != GL_WAIT_FAILED) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, slot->pbo);
            const size_t bytes = size_t(slot->width) * slot->height * 4;
            if (const auto* pixels = static_cast<const uint8_t*>(glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes, GL_MAP_READ_BIT))) {
                frame.rgba.assign(pixels, pixels + bytes);
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            }
        }
        glDeleteSync(slot->fence);
        slot->fence = nullptr;
        if (frame.rgba.empty()) {
            ++counts.glErrors;
            continue;
        }
        ++counts.completed;
        done(std::move(frame));
    }
}

void Thumbnail::destroy() {
    for (auto& slot : slots) {
        if (slot.fence)
            glDeleteSync(slot.fence);
        if (slot.pbo)
            glDeleteBuffers(1, &slot.pbo);
        slot = {};
    }
    if (program)
        glDeleteProgram(program);
    if (image)
        glDeleteTextures(1, &image);
    if (framebuffer)
        glDeleteFramebuffers(1, &framebuffer);
    if (sampler)
        glDeleteSamplers(1, &sampler);
    program = image = framebuffer = sampler = 0;
    initTried = false;
    failureText.clear();
}

const Thumbnail::SCounters& Thumbnail::counters() {
    return counts;
}

const char* Thumbnail::failure() {
    return failureText.c_str();
}
