# Builds build/hyprmorph.so: the Hyprland plugin with the prep-worker executable linked in (run from a memfd at load), so an install
# is the .so alone. Needs Hyprland headers (pkg-config hyprland), hyprutils, libdrm, EGL/GLESv2/GBM and OpenCV 5 (worker only).
CXX ?= c++

PREP_CXXFLAGS := -O2 -std=c++20 -Wall -Wextra -ffp-contract=off $(shell pkg-config --cflags opencv5 egl glesv2 gbm)
PREP_LIBS     := $(shell pkg-config --libs egl glesv2 gbm) -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_features -lopencv_geometry
PREP_OBJ      := $(patsubst prep/src/%.cpp,build/prep/%.o,$(wildcard prep/src/*.cpp))
PREP_HEADERS  := $(wildcard prep/src/*.hpp)

PLUGIN_SOURCES := $(wildcard plugin/*.cpp) prep/src/wire.cpp prep/src/types.cpp

.PHONY: all clean
all: build/hyprmorph.so

build/prep/%.o: prep/src/%.cpp $(PREP_HEADERS)
	mkdir -p build/prep
	$(CXX) $(PREP_CXXFLAGS) -c $< -o $@

build/prep/libprep.a: $(PREP_OBJ)
	ar rcs $@ $^

build/prep-worker: prep/worker/prep_worker.cpp build/prep/libprep.a
	$(CXX) $(PREP_CXXFLAGS) $< build/prep/libprep.a -o $@ $(PREP_LIBS)

build/readiness.o: readiness/readiness.cpp readiness/readiness.hpp
	mkdir -p build
	$(CXX) -O3 -std=c++20 -Wall -Wextra -Wpedantic -fPIC -c $< -o $@

# -fno-gnu-unique: header-inline globals (Log::logger, ...) bind to Hyprland's instances and dlclose can unmap the plugin; as GNU unique
# symbols the .so becomes NODELETE and a reload keeps running the old build.
# HOLD_HYPRUTILS_VERSION: Hold refuses to hook a libhyprutils other than the one built against.
build/hyprmorph.so: $(PLUGIN_SOURCES) $(wildcard plugin/*.hpp) readiness/readiness.hpp $(PREP_HEADERS) build/readiness.o build/prep-worker
	$(CXX) -O2 -std=c++23 -Wall -Wextra -fno-gnu-unique -fPIC -shared $(PLUGIN_SOURCES) build/readiness.o -o $@ \
		-DHOLD_HYPRUTILS_VERSION='"$(shell pkg-config --modversion hyprutils)"' -DPREP_WORKER_BIN='"build/prep-worker"' \
		$(shell pkg-config --cflags hyprland libdrm) $(shell pkg-config --libs glesv2 libdrm)

clean:
	rm -rf build
