#pragma once
// The prep-worker executable, linked into the plugin as bytes (EmbeddedWorker.cpp, .incbin of build/prep-worker). Prep runs it from a memfd.
#include <cstddef>

namespace EmbeddedWorker {
    const void* begin();
    size_t      size();
}
