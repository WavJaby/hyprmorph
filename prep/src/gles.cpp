#include "gles.hpp"

#include <fcntl.h>
#include <unistd.h>

namespace prep::gles {

Context::Context(const std::string& node) {
    fd = open(node.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("cannot open render node " + node);
    gbm = gbm_create_device(fd);
    if (!gbm) throw std::runtime_error("cannot create GBM device");
    auto getDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (!getDisplay) throw std::runtime_error("no EGL GBM platform");
    display = getDisplay(EGL_PLATFORM_GBM_KHR, gbm, nullptr);
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, nullptr, nullptr) || !eglBindAPI(EGL_OPENGL_ES_API))
        throw std::runtime_error("cannot initialize EGL GLES");
    // Surfaceless: compute only, no config needed when EGL_KHR_no_config_context is present; fall back to one config.
    EGLConfig config = EGL_NO_CONFIG_KHR;
    const char* extensions = eglQueryString(display, EGL_EXTENSIONS);
    if (!extensions || !std::strstr(extensions, "EGL_KHR_no_config_context")) {
        EGLint count = 0;
        constexpr EGLint attrs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE};
        if (!eglChooseConfig(display, attrs, &config, 1, &count) || count != 1) throw std::runtime_error("no GLES3 config");
    }
    constexpr EGLint ctxAttrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION_KHR, 2, EGL_NONE};
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctxAttrs);
    if (context == EGL_NO_CONTEXT || !eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context))
        throw std::runtime_error("cannot make GLES 3.2 context current");
}

Context::~Context() {
    if (display != EGL_NO_DISPLAY) {
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
        eglTerminate(display);
    }
    if (gbm) gbm_device_destroy(gbm);
    if (fd >= 0) close(fd);
}

void check(const char* step) {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) throw std::runtime_error(std::string(step) + ": GL error " + std::to_string(error));
}

GLuint program(const char* source) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(size_t(length), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        throw std::runtime_error("compute shader: " + log);
    }
    GLuint id = glCreateProgram();
    glAttachShader(id, shader);
    glLinkProgram(id);
    glDeleteShader(shader);
    glGetProgramiv(id, GL_LINK_STATUS, &ok);
    if (!ok) throw std::runtime_error("cannot link compute program");
    return id;
}

void Buffer::reserve(size_t bytes) {
    if (!id) glGenBuffers(1, &id);
    if (bytes <= capacity) return;
    capacity = std::max<size_t>(bytes, 2 * capacity);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, GLsizeiptr(capacity), nullptr, GL_DYNAMIC_DRAW);
}

void Buffer::upload(const void* data, size_t bytes, size_t offset) {
    reserve(offset + bytes);
    if (!bytes) return;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, GLintptr(offset), GLsizeiptr(bytes), data);
}

void Buffer::download(void* data, size_t bytes, size_t offset) const {
    if (!bytes) return;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
    void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, GLintptr(offset), GLsizeiptr(bytes), GL_MAP_READ_BIT);
    if (!mapped) throw std::runtime_error("cannot map storage buffer");
    std::memcpy(data, mapped, bytes);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

Buffer::~Buffer() {
    if (id) glDeleteBuffers(1, &id);
}

} // namespace prep::gles
