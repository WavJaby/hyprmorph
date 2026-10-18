#pragma once
// Headless EGL (GBM platform) + GLES 3.2 compute on one render node, owned by the prep worker (never the compositor).
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <gbm.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace prep::gles {

struct Context {
    int fd = -1;
    gbm_device* gbm = nullptr;
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    explicit Context(const std::string& node);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
};

void check(const char* step);
// Compile + link one compute shader; throws with the info log.
GLuint program(const char* source);

// Shader storage buffer that grows on demand (contents are not preserved across growth).
struct Buffer {
    GLuint id = 0;
    size_t capacity = 0;
    void reserve(size_t bytes);
    void upload(const void* data, size_t bytes, size_t offset = 0);
    void download(void* data, size_t bytes, size_t offset = 0) const;
    void bind(GLuint binding) const { glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, id); }
    ~Buffer();
};

} // namespace prep::gles
