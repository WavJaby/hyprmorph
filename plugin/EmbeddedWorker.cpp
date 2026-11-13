#include "EmbeddedWorker.hpp"

#ifndef PREP_WORKER_BIN
#error "PREP_WORKER_BIN must name the prep-worker executable to embed"
#endif

// Hidden: two loaded copies of the plugin (a reload under a new path) must not resolve each other's image.
asm(".section .rodata\n"
    ".balign 16\n"
    ".hidden rt_prep_worker_begin\n"
    ".hidden rt_prep_worker_end\n"
    "rt_prep_worker_begin:\n"
    ".incbin \"" PREP_WORKER_BIN "\"\n"
    "rt_prep_worker_end:\n"
    ".previous\n");

extern "C" const char rt_prep_worker_begin[], rt_prep_worker_end[];

const void* EmbeddedWorker::begin() {
    return rt_prep_worker_begin;
}

size_t EmbeddedWorker::size() {
    return size_t(rt_prep_worker_end - rt_prep_worker_begin);
}
