# Source files
# SRC := 000-sanity-check.cu
# SRC := 001-raw-fp8-mm.cu
# SRC := 002-fp8-mm.cu
# SRC := 003-half-raw-fp8-mm.cu
# SRC := 004-tm-ldst-1d-int32.cu
# SRC := 005-tm-alloc.cu
# SRC := 006-tm-ldst-2d-int32.cu
# SRC := 007-tm-cp.cu
# SRC := 008-tm-cp-fp8.cu
# SRC := 009-mxfp8-mm.cu
# SRC := 010-mxfp8-mma.cu
# SRC := 011-fp8-mma-wg-ablation.cu
# SRC := 012-fp8-mma-N-ablation.cu
# SRC := 013-fp8-mma-M-ablation.cu
# SRC := 014-fp8-mma-M-ablation-2.cu
# SRC := 015-fp8-mma-M-ablation-3.cu
# SRC := 016-mxfp8-mma-polished.cu
# SRC := 017-mxfp8-mma-M256.cu
# SRC := 018-mxfp8-quantize.cu
# SRC := 019-smem-swizzle-8b.cu
# SRC := 020-smem-swizzle-16b.cu
# SRC := 021-mxfp8-quantize-opt.cu
# SRC := 022-mxfp8-quantize-opt-2.cu
# SRC := 023-mxfp8-mma-grid-ablation.cu
# SRC := 024-mxfp8-quantize-grid-ablation.cu
# SRC := 027-raw-binding.cu
# SRC := 028-tma-load.cu
# SRC := 029-tma-store.cu
# SRC := 030-tma-load-bandwidth.cu
# SRC := 031-tma-load-store-bandwidth.cu
# SRC := 032-tma-swiz-load.cu
# SRC := 033-tma-swiz-load-bandwidth.cu
# SRC := 034-dma-bandwidth.cu
# SRC := 035-tma-pipeline-bandwidth.cu
# SRC := 036-mxfp8-quantize-opt-3.cu
# SRC := 037-mxfp8-transpose-quantize.cu
# SRC := 038-mxfp8-quantize-combined.cu
# SRC := 039-fp8-mma-2cta.cu
# SRC := 040-fp8-mma-2cta-factored.cu
# SRC := 041-fp8-mma-2cta-M256.cu
# SRC := 042-fp8-mma-2cta-MN128.cu
# SRC := 043-fp8-mma-2cta-MNK128.cu
# SRC := 044-mxfp8-mm-2cta.cu
# SRC := 045-mxfp8-mma-2cta.cu
# SRC := 046-mxfp8-mma-2cta-MNK128.cu
# SRC := 047-mxfp8-mma-2cta-MNK128-polished.cu
# SRC := 048-bf16-matmul-shape-ablation.cu
# SRC := 049-bf16-matmul-launcher-ablation.cu
# SRC := 050-bf16-mha-D128.cu
# SRC := 051-bf16-mha-opt-D128.cu
# SRC := 052-bf16-mha-dsv.cu
# SRC := 053-microbenchmarks.cu
# SRC := 054-multi-node-ib.cu
# SRC := 057-tcgen05-bf16-mma-smallM.cu
# SRC := 058-tensor-ipc.cu
# SRC := 059-tensor-ipc-polished.cu
# SRC := 060-all2all.cu
# SRC := 061-vmm-ipc.cu
# SRC := 062-vmm-ipc-polished.cu
# SRC := 063-parallel-tensor.cu
# SRC := 064-multigpu-sync.cu
# SRC := 065-multimem-all-reduce.cu
# SRC := 067-tma-vs-warp-1.cu
# SRC := 068-tma-vs-warp-2.cu
# SRC := 069-multimem-all-reduce-tma.cu
# SRC := 070-nvlink-bandwidth-copy-engine.cu
# SRC := 071-nvlink-bandwidth-sm.cu
# SRC := 072-nvlink-bandwidth-sm-tma.cu
# SRC := 073-atomic-check.cu
# SRC := 074-faster-comms-sms.cu
# SRC := 075-nvshmem.cu
# SRC := 076-nvshmem-all-reduce.cu
# SRC := 081-tma-load-store-irregular.cu
# SRC := 083-nvfp4-gemm.cu
# SRC := 084-smem-swizzle-atom.cu
# SRC := 085-smem-swizzle.cu
# SRC := 086-clc.cu
# SRC := 087-pure-mma-speed.cu
# SRC := 088-pure-tma-speed.cu
# SRC := 089-clc-multicast.cu
# SRC := 090-2cw-4cg-multicast.cu
# SRC := 091-smem-mma-tmem-pipe.cu
# SRC := 092-elect-sync.cu
# SRC := 093-cutlass-bf16-gemm.cu
# SRC := 094-l2-cache-throughput.cu
# SRC := 095-l2-cache-throughput-pg.cu
# SRC := 096-l2-cache-throughput-pg-numa.cu
# SRC := 097-threadblock-allocation.cu
# SRC := 098-l2-cache-throughput-ss.cu
# SRC := 099-l2-cache-throughput-ss-allocator.cu
# SRC := 100-l2-cache-partition-check.cu
# SRC := 101-l2-cache-partition-check-2.cu
SRC := 102-pdl-ordering-check.cu

# Binaries
CC := g++
NVCC := nvcc
MPICC := mpicxx
MPIRUN := mpirun

# ThunderKittens configuration
THUNDERKITTENS_ROOT := ../ThunderKittens

# Python binding configuration
# These take a while to load. You can bypass them by defining them as environment variables in advance
PYTHON_VERSION ?= $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_INCLUDES ?= $(shell python3 -c "import sysconfig; print('-I', sysconfig.get_path('include'), sep='')")
PYBIND_INCLUDES ?= $(shell python3 -m pybind11 --includes)
PYTORCH_INCLUDES ?= $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTHON_LIBDIR ?= $(shell python3 -c "import sysconfig; print('-L', sysconfig.get_config_var('LIBDIR'), sep='')")
PYTORCH_LIBDIR ?= $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")

# NVCC flags (ours)
NVCCFLAGS := -DNDEBUG -lineinfo
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr 
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing 
NVCCFLAGS += -forward-unknown-to-host-compiler -ftemplate-backtrace-limit=0
NVCCFLAGS += -std=c++20 -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static
NVCCFLAGS += -O3 --use_fast_math # VERY important to include this flag
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills 
NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype
NVCCFLAGS+= -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a

# Choose one of the following configurations:

# 1. Standalone configuration
OUT := $(basename $(SRC)).out
RUN_CMD := ./$(OUT)

# 2. Cutlass configuration
# OUT := $(basename $(SRC)).out
# CUTLASS_ROOT ?= /home/stuart/stuart/cutlass
# NVCCFLAGS += -I$(CUTLASS_ROOT)/include/ -I$(CUTLASS_ROOT)/tools/util/include/
# NVCCFLAGS += -diag-suppress 2908
# RUN_CMD := ./$(OUT)

# 3. Python binding configuration
# NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
# NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
# NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H
# NVCCFLAGS += $(PYTHON_INCLUDES) $(PYBIND_INCLUDES)
# NVCCFLAGS += -diag-suppress 3189 # for warnings generated by PyTorch C++ headers
# NVCCFLAGS += -shared -fPIC
# NVCCFLAGS += ${PYTHON_LIBDIR} ${PYTORCH_LIBDIR} -lpython${PYTHON_VERSION}
# NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10
# OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# RUN_CMD := python3 $(basename $(SRC)).py

# 4. PyTorch binding configuration
# NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
# NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
# NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H
# NVCCFLAGS += $(PYTHON_INCLUDES) $(PYTORCH_INCLUDES) $(PYBIND_INCLUDES)
# NVCCFLAGS += -diag-suppress 3189 # for warnings generated by PyTorch C++ headers
# NVCCFLAGS += -shared -fPIC
# NVCCFLAGS += ${PYTHON_LIBDIR} ${PYTORCH_LIBDIR} -lpython${PYTHON_VERSION}
# NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10
# NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C # replicate setup.py behavior
# OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# RUN_CMD := python3 $(basename $(SRC)).py

# 5. PyTorch + Torchrun binding configuration
# NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
# NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
# NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H
# NVCCFLAGS += $(PYTHON_INCLUDES) $(PYTORCH_INCLUDES) $(PYBIND_INCLUDES)
# NVCCFLAGS += -diag-suppress 3189 # for warnings generated by PyTorch C++ headers
# NVCCFLAGS += -shared -fPIC
# NVCCFLAGS += ${PYTHON_LIBDIR} ${PYTORCH_LIBDIR} -lpython${PYTHON_VERSION}
# NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10
# NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C # replicate setup.py behavior
# OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# RUN_CMD := OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 $(basename $(SRC)).py

# 6. MPI configuration
# NVCCFLAGS += -ccbin ${MPICC}
# OUT := $(basename $(SRC)).out
# RUN_CMD := $(MPIRUN) -np 8 ./$(OUT)

# 7. NVSHMEM configuration
# NVCCFLAGS += -ccbin ${MPICC}
# NVCCFLAGS += -rdc=true
# NVCCFLAGS += -I/usr/include/nvshmem_12/ -L/usr/lib/x86_64-linux-gnu/nvshmem/12 -lnvshmem_host -lnvshmem_device
# NVCCFLAGS += -diag-suppress 3013 -diag-suppress 3012
# OUT := $(basename $(SRC)).out
# RUN_CMD := $(MPIRUN) -np 8 ./$(OUT)

all: $(OUT)

ptx: $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) --ptx -o $(basename $(SRC)).ptx

run: $(OUT)
	$(RUN_CMD)

$(OUT): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(OUT)

clean:
	rm -f $(OUT) $(basename $(SRC)).ptx $(basename $(SRC)).out
