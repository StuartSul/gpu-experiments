SRC := jitkittens.cpp
PYTHON ?= python3
CUDA_HOME ?= /usr/local/cuda

PYBIND_INCLUDES ?= $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXTENSION_SUFFIX ?= $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT := _C$(PYTHON_EXTENSION_SUFFIX)

COMPILE_FLAGS := $(PYBIND_INCLUDES) -I$(CUDA_HOME)/include
COMPILE_FLAGS += -O3 -DNDEBUG -std=c++20
COMPILE_FLAGS += -Wno-psabi -fno-strict-aliasing -ftemplate-backtrace-limit=0
COMPILE_FLAGS += -fPIC -shared -L$(CUDA_HOME)/lib64 -lnvrtc -lcuda

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $< $(COMPILE_FLAGS) -o $@

clean:
	rm -f $(OUT)
