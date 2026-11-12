#include "Presenter.hpp"

#include <GLES3/gl31.h>

#include <array>
#include <cstddef>
#include <string>

namespace {
    GLuint      program = 0, vao = 0, vbo = 0, ibo = 0, sampler = 0;
    GLint       uExtent = -1, uOpacity = -1, uSource = -1;
    bool        initTried = false;
    std::string failureText;

    // Hyprland caches GL state; everything touched here is put back.
    struct SSavedState {
        GLint     program, activeTexture, texture, boundSampler, drawFramebuffer, vertexArray, arrayBuffer, blendSrcRgb, blendDstRgb, blendSrcAlpha, blendDstAlpha;
        GLint     viewport[4], scissorBox[4];
        GLfloat   clearColor[4];
        GLboolean blend, scissor, depth, stencil, cull, colorMask[4];
        SSavedState() {
            glGetIntegerv(GL_CURRENT_PROGRAM, &program);
            glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
            glActiveTexture(GL_TEXTURE0);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &texture);
            glGetIntegerv(GL_SAMPLER_BINDING, &boundSampler);
            glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFramebuffer);
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vertexArray);
            glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &arrayBuffer);
            glGetIntegerv(GL_BLEND_SRC_RGB, &blendSrcRgb);
            glGetIntegerv(GL_BLEND_DST_RGB, &blendDstRgb);
            glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrcAlpha);
            glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDstAlpha);
            glGetIntegerv(GL_VIEWPORT, viewport);
            glGetIntegerv(GL_SCISSOR_BOX, scissorBox);
            glGetFloatv(GL_COLOR_CLEAR_VALUE, clearColor);
            glGetBooleanv(GL_COLOR_WRITEMASK, colorMask);
            blend   = glIsEnabled(GL_BLEND);
            scissor = glIsEnabled(GL_SCISSOR_TEST);
            depth   = glIsEnabled(GL_DEPTH_TEST);
            stencil = glIsEnabled(GL_STENCIL_TEST);
            cull    = glIsEnabled(GL_CULL_FACE);
        }
        ~SSavedState() {
            const auto set = [](GLenum cap, GLboolean on) { on ? glEnable(cap) : glDisable(cap); };
            set(GL_BLEND, blend);
            set(GL_SCISSOR_TEST, scissor);
            set(GL_DEPTH_TEST, depth);
            set(GL_STENCIL_TEST, stencil);
            set(GL_CULL_FACE, cull);
            glColorMask(colorMask[0], colorMask[1], colorMask[2], colorMask[3]);
            glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
            glScissor(scissorBox[0], scissorBox[1], scissorBox[2], scissorBox[3]);
            glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
            glBlendFuncSeparate(blendSrcRgb, blendDstRgb, blendSrcAlpha, blendDstAlpha);
            glBindVertexArray(vertexArray); // the element buffer binding belongs to the VAO
            glBindBuffer(GL_ARRAY_BUFFER, arrayBuffer);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFramebuffer);
            glBindSampler(0, boundSampler);
            glBindTexture(GL_TEXTURE_2D, texture);
            glActiveTexture(activeTexture);
            glUseProgram(program);
        }
    };

    GLuint compile(GLenum type, const char* source) {
        const GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            std::array<char, 1024> log{};
            glGetShaderInfoLog(shader, log.size(), nullptr, log.data());
            failureText = std::string{"presenter shader: "} + log.data();
            glDeleteShader(shader);
            return 0;
        }
        return shader;
    }

    bool init() {
        initTried                    = true;
        constexpr const char* VERTEX = R"(#version 310 es
            precision highp float;
            layout(location=0) in vec2 position;
            layout(location=1) in vec2 coordinate;
            uniform vec2 extent;
            out vec2 uv;
            void main() {
                // No flip: framebuffer row 0 is the top of the content, in the target as in the sources.
                gl_Position = vec4(2.0 * position - 1.0, 0.0, 1.0);
                uv = coordinate * extent;
            })";
        constexpr const char* FRAGMENT = R"(#version 310 es
            precision highp float;
            uniform sampler2D sourceImage;
            uniform float opacity;
            in vec2 uv;
            layout(location=0) out vec4 color;
            void main() {
                color = texture(sourceImage, uv) * opacity;
            })";
        const GLuint vs = compile(GL_VERTEX_SHADER, VERTEX), fs = vs ? compile(GL_FRAGMENT_SHADER, FRAGMENT) : 0;
        if (!vs || !fs) {
            if (vs)
                glDeleteShader(vs);
            return false;
        }
        program = glCreateProgram();
        glAttachShader(program, vs);
        glAttachShader(program, fs);
        glLinkProgram(program);
        glDeleteShader(vs);
        glDeleteShader(fs);
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            failureText = "presenter program link failed";
            return false;
        }
        uExtent  = glGetUniformLocation(program, "extent");
        uOpacity = glGetUniformLocation(program, "opacity");
        uSource  = glGetUniformLocation(program, "sourceImage");

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ibo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Geometry::SVertex), reinterpret_cast<void*>(offsetof(Geometry::SVertex, x)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Geometry::SVertex), reinterpret_cast<void*>(offsetof(Geometry::SVertex, u)));
        glGenSamplers(1, &sampler);
        glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        return true;
    }
}

bool Presenter::compose(unsigned framebuffer, int width, int height, std::span<const SLayer> layers) {
    if (!failureText.empty())
        return false;
    SSavedState saved;
    if (!initTried && !init())
        return false;

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
    glViewport(0, 0, width, height);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_SCISSOR_TEST);
    glScissor(0, 0, width, height);
    glClearColor(0.F, 0.F, 0.F, 0.F);
    glClear(GL_COLOR_BUFFER_BIT);
    // Sum of opacity-weighted layers; the cleared target makes the first layer a plain copy.
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);

    glUseProgram(program);
    glUniform1i(uSource, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindSampler(0, sampler);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    for (const auto& layer : layers) {
        if (!layer.batch || layer.batch->indices.empty() || layer.opacity <= 0.F)
            continue;
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(layer.batch->vertices.size() * sizeof(Geometry::SVertex)), layer.batch->vertices.data(), GL_STREAM_DRAW);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, GLsizeiptr(layer.batch->indices.size() * sizeof(uint32_t)), layer.batch->indices.data(), GL_STREAM_DRAW);
        glBindTexture(GL_TEXTURE_2D, layer.texture);
        glUniform2f(uExtent, layer.uExtent, layer.vExtent);
        glUniform1f(uOpacity, layer.opacity);
        glDrawElements(GL_TRIANGLES, GLsizei(layer.batch->indices.size()), GL_UNSIGNED_INT, nullptr);
    }
    if (glGetError() != GL_NO_ERROR) {
        failureText = "presenter draw raised a GL error";
        return false;
    }
    return true;
}

void Presenter::destroy() {
    if (program)
        glDeleteProgram(program);
    if (vao)
        glDeleteVertexArrays(1, &vao);
    if (vbo)
        glDeleteBuffers(1, &vbo);
    if (ibo)
        glDeleteBuffers(1, &ibo);
    if (sampler)
        glDeleteSamplers(1, &sampler);
    program = vao = vbo = ibo = sampler = 0;
    initTried = false;
    failureText.clear();
}

const char* Presenter::failure() {
    return failureText.c_str();
}
