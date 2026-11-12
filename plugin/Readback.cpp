#include "Readback.hpp"

#include <GLES3/gl31.h>

#include <algorithm>
#include <array>

namespace {
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

    GLsync fenceOf(void* p) {
        return static_cast<GLsync>(p);
    }
}

Readback::CPool::CPool(size_t slots, int minWidth, int minHeight) : m_slots(slots), m_minW(minWidth), m_minH(minHeight) {}

bool Readback::CPool::init() {
    m_initTried                  = true;
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
        m_failure = std::string{"compute shader: "} + log.data();
        glDeleteShader(shader);
        return false;
    }
    m_program = glCreateProgram();
    glAttachShader(m_program, shader);
    glLinkProgram(m_program);
    glDeleteShader(shader);
    glGetProgramiv(m_program, GL_LINK_STATUS, &ok);
    if (!ok) {
        m_failure = "compute program link failed";
        return false;
    }
    m_uSource = glGetUniformLocation(m_program, "sourceImage");
    m_uExtent = glGetUniformLocation(m_program, "extent");
    m_uCrop   = glGetUniformLocation(m_program, "crop");

    glGenFramebuffers(1, &m_framebuffer);
    glGenSamplers(1, &m_sampler);
    glSamplerParameteri(m_sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(m_sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(m_sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(m_sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    for (auto& slot : m_slots)
        glGenBuffers(1, &slot.pbo);
    return true;
}

bool Readback::CPool::reserve(int width, int height) {
    if (width <= m_capacityW && height <= m_capacityH)
        return true;
    if (!m_failure.empty())
        return false;
    SSavedState saved;
    return (m_initTried || init()) && grow(width, height);
}

// Storage for width x height. Replaces the target image and re-specifies every PBO, so it waits until no readback is in flight.
bool Readback::CPool::grow(int width, int height) {
    if (width <= m_capacityW && height <= m_capacityH)
        return true;
    if (std::ranges::any_of(m_slots, [](const SSlot& s) { return s.fence; }))
        return false;
    const int w = std::max({width, m_capacityW, m_minW}), h = std::max({height, m_capacityH, m_minH});
    if (m_image)
        glDeleteTextures(1, &m_image);
    glGenTextures(1, &m_image);
    glBindTexture(GL_TEXTURE_2D, m_image);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_framebuffer);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_image, 0);
    if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        m_failure = "readback framebuffer incomplete";
        return false;
    }
    for (auto& slot : m_slots) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, slot.pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, GLsizeiptr(w) * h * 4, nullptr, GL_STREAM_READ);
    }
    m_capacityW = w;
    m_capacityH = h;
    return true;
}

uint64_t Readback::CPool::submit(unsigned texture, float u0, float v0, float uw, float vh, int width, int height) {
    if (!m_failure.empty() || width < 1 || height < 1)
        return 0;
    const auto free = std::ranges::find_if(m_slots, [](const SSlot& s) { return !s.fence; });
    if (free == m_slots.end()) {
        ++m_counts.busy;
        return 0;
    }
    SSavedState saved;
    if (!m_initTried && !init())
        return 0;
    if (!grow(width, height)) {
        ++m_counts.busy;
        return 0;
    }

    glUseProgram(m_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glBindSampler(0, m_sampler);
    glBindImageTexture(0, m_image, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    glUniform1i(m_uSource, 0);
    glUniform2i(m_uExtent, width, height);
    glUniform4f(m_uCrop, u0, v0, uw, vh);
    glDispatchCompute((width + 7) / 8, (height + 7) / 8, 1);
    glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_framebuffer);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, free->pbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    free->fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
    if (glGetError() != GL_NO_ERROR)
        ++m_counts.glErrors;
    free->ticket = ++m_tickets;
    free->width  = width;
    free->height = height;
    ++m_counts.submitted;
    return free->ticket;
}

void Readback::CPool::poll(const std::function<void(uint64_t, int, int, std::vector<uint8_t>&&)>& done) {
    std::vector<SSlot*> ordered;
    for (auto& slot : m_slots)
        if (slot.fence)
            ordered.push_back(&slot);
    if (ordered.empty())
        return;
    std::ranges::sort(ordered, {}, &SSlot::ticket);

    SSavedState saved;
    for (auto* slot : ordered) {
        const auto status = glClientWaitSync(fenceOf(slot->fence), 0, 0);
        if (status == GL_TIMEOUT_EXPIRED)
            break;
        std::vector<uint8_t> rgba;
        if (status != GL_WAIT_FAILED) {
            glBindBuffer(GL_PIXEL_PACK_BUFFER, slot->pbo);
            const size_t bytes = size_t(slot->width) * slot->height * 4;
            if (const auto* pixels = static_cast<const uint8_t*>(glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes, GL_MAP_READ_BIT))) {
                rgba.assign(pixels, pixels + bytes);
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            }
        }
        glDeleteSync(fenceOf(slot->fence));
        slot->fence = nullptr;
        if (rgba.empty()) {
            ++m_counts.glErrors;
            continue;
        }
        ++m_counts.completed;
        done(slot->ticket, slot->width, slot->height, std::move(rgba));
    }
}

void Readback::CPool::destroy() {
    for (auto& slot : m_slots) {
        if (slot.fence)
            glDeleteSync(fenceOf(slot.fence));
        if (slot.pbo)
            glDeleteBuffers(1, &slot.pbo);
        slot = {};
    }
    if (m_program)
        glDeleteProgram(m_program);
    if (m_image)
        glDeleteTextures(1, &m_image);
    if (m_framebuffer)
        glDeleteFramebuffers(1, &m_framebuffer);
    if (m_sampler)
        glDeleteSamplers(1, &m_sampler);
    m_program = m_image = m_framebuffer = m_sampler = 0;
    m_capacityW = m_capacityH = 0;
    m_initTried               = false;
    m_failure.clear();
}
