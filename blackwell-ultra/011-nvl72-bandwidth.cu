#include "kittens.cuh"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace kittens;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 2;
    static constexpr int ROW_BLOCK_SIZE = 128;
    static constexpr int COL_BLOCK_SIZE = 128;

    using tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, 1, -1, -1, -1, tile>, NUM_DEVICES, false>;

    parallel_layout tensor;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(tensor.cols() / COL_BLOCK_SIZE, tensor.rows() / ROW_BLOCK_SIZE, tensor.depth());
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(tile) + 1024);
    }
};

// Kernel to initialize memory with a value
__global__ void initKernel(float* data, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        data[i] = value;
    }
}

// Kernel to verify data correctness
__global__ void verifyKernel(float* data, float expected, size_t n, int* errorCount) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        if (fabsf(data[i] - expected) > 1e-5f) {
            atomicAdd(errorCount, 1);
        }
    }
}

__global__ void copyKernel(float* dst, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__launch_bounds__(config::NUM_THREADS, 6)
__global__ void tma_copy_kernel(const __grid_constant__ globals G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::tile &tile = allocator.allocate<globals::tile>();

    const int depth_idx = blockIdx.z;
    const int row_block_idx = blockIdx.y;
    const int col_block_idx = blockIdx.x;

    __shared__ semaphore inputs_arrived;
    init_semaphore(inputs_arrived, 0, 1);
    tma::expect_bytes(inputs_arrived, sizeof(globals::tile));
    tma::load_async(tile, G.tensor[0], {depth_idx, row_block_idx, col_block_idx}, inputs_arrived);
    wait(inputs_arrived, 0);
    tma::store_async(G.tensor[1], tile, {depth_idx, row_block_idx, col_block_idx});
}

// NVL72 copy-engine / TMA / SM-copy bandwidth
//   mode 0 -> copy engine     (cudaMemcpyAsync, like 000-nvlink-bandwidth-copy-engine.cu)
//   mode 1 -> TMA             (tma load/store,  like 001-nvlink-bandwidth-sm-tma.cu)
//   mode 2 -> SM ld/st        (dst[i]=src[i],   like 002-nvlink-bandwidth-sm.cu)
static double benchmark(uintptr_t dst_ptr, uintptr_t src_ptr, size_t nbytes,
                        int warmup_iters, int timed_iters, int mode) {
    float* src_data = reinterpret_cast<float*>(src_ptr);
    float* dst_data = reinterpret_cast<float*>(dst_ptr);
    const size_t numElements = nbytes / sizeof(float);
    const float srcValue = 3.14f;
    const float dstInitValue = 0.0f;

    // Initialize source with 3.14 and destination with 0.
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    // Limit grid size to avoid excessive blocks
    gridSize = min(gridSize, 65536);

    initKernel<<<gridSize, blockSize>>>(src_data, srcValue, numElements);
    initKernel<<<gridSize, blockSize>>>(dst_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());

    cudaStream_t stream;
    cudaEvent_t start, stop;
    CUDACHECK(cudaStreamCreate(&stream));
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    constexpr size_t TMA_SLICE = 32768;
    bf16* tma_data[2] = {reinterpret_cast<bf16*>(src_data), reinterpret_cast<bf16*>(dst_data)};
    const size_t tma_depth = nbytes / (TMA_SLICE * TMA_SLICE * sizeof(bf16));
    globals G {
        .tensor = globals::parallel_layout{tma_data, nullptr, tma_depth, TMA_SLICE, TMA_SLICE},
        .dev_idx = 0
    };
    const dim3 tma_grid = G.grid();
    const int tma_smem = G.dynamic_shared_memory();

    auto run = [&]() {
        if (mode == 0) {
            CUDACHECK(cudaMemcpyAsync(dst_data, src_data, nbytes, cudaMemcpyDeviceToDevice, stream));
        } else if (mode == 1) {
            tma_copy_kernel<<<tma_grid, config::NUM_THREADS, tma_smem, stream>>>(G);
        } else {
            copyKernel<<<gridSize, blockSize, 0, stream>>>(dst_data, src_data, numElements);
        }
    };

    // Warm-up transfer(s).
    for (int i = 0; i < warmup_iters; ++i) run();
    CUDACHECK(cudaStreamSynchronize(stream));

    // Timed transfer(s): source -> destination, measured with CUDA events.
    CUDACHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < timed_iters; ++i) run();
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    float milliseconds = 0;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Verify correctness on the destination.
    int* d_errorCount;
    CUDACHECK(cudaMalloc(&d_errorCount, sizeof(int)));
    CUDACHECK(cudaMemset(d_errorCount, 0, sizeof(int)));

    verifyKernel<<<gridSize, blockSize>>>(dst_data, srcValue, numElements, d_errorCount);
    CUDACHECK(cudaDeviceSynchronize());

    int h_errorCount;
    CUDACHECK(cudaMemcpy(&h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaFree(d_errorCount));

    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    CUDACHECK(cudaStreamDestroy(stream));

    if (h_errorCount != 0)
        throw std::runtime_error("correctness check FAILED (mode " + std::to_string(mode) +
                                 "): " + std::to_string(h_errorCount) + " mismatches");

    return static_cast<double>(milliseconds) / timed_iters;
}

PYBIND11_MODULE(_C, m) {
    m.def("benchmark", &benchmark, "",
          py::arg("dst_ptr"), py::arg("src_ptr"), py::arg("nbytes"),
          py::arg("warmup_iters") = 1, py::arg("timed_iters") = 1, py::arg("mode") = 0);
}
