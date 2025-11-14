/*
    Transfer time: 14.946 ms
    TMA Bandwidth: 669.09 GB/s
*/

#include "kittens.cuh"

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

// Kernel to initialize bf16 memory with a value
__global__ void initKernel(bf16* data, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        data[i] = __float2bfloat16(value);
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

// Kernel to verify data correctness
__global__ void verifyKernel(bf16* data, float expected, size_t n, int* errorCount) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        float val = __bfloat162float(data[i]);
        if (fabsf(val - expected) > 1e-2f) {
            atomicAdd(errorCount, 1);
        }
    }
}

int main() {
    // Configuration
    const size_t dataSize = 10ULL * 1024 * 1024 * 1024;  // 10 GB
    const size_t numElements = dataSize / sizeof(bf16);
    const float srcValue = 3.14f;
    const float dstInitValue = 0.0f;
    
    printf("NVLink Bandwidth Test (TMA-based Copy)\n");
    printf("========================================\n");
    printf("Data size: %.2f GB\n", dataSize / (1024.0 * 1024.0 * 1024.0));
    printf("Number of bf16 elements: %zu\n", numElements);
    printf("Using ThunderKittens TMA primitives\n\n");
    
    // Enable peer access
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));
    printf("Peer access enabled between devices\n\n");

    // Allocate memory on device 0
    bf16* d0_data;
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMalloc(&d0_data, dataSize));
    printf("Allocated %.2f GB on Device 0\n", dataSize / (1024.0 * 1024.0 * 1024.0));

    // Initialize device 0 memory with 3.14
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    gridSize = min(gridSize, 65536);

    initKernel<<<gridSize, blockSize>>>(d0_data, srcValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    printf("Initialized Device 0 memory with value: %.2f\n", srcValue);
    
    // Allocate memory on device 1
    bf16* d1_data;
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMalloc(&d1_data, dataSize));
    printf("Allocated %.2f GB on Device 1\n", dataSize / (1024.0 * 1024.0 * 1024.0));
    
    // Initialize device 1 memory with 0
    initKernel<<<gridSize, blockSize>>>(d1_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    printf("Initialized Device 1 memory with value: %.2f\n\n", dstInitValue);
    
    // Create events for timing
    CUDACHECK(cudaSetDevice(0));
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    
    // Create globals
    bf16 *data[2] = {d0_data, d1_data};
    globals G {
        .tensor = globals::parallel_layout{data, nullptr, 5, 32768, 32768},
        .dev_idx = 0
    };

    // Warm up run
    printf("Performing warm-up TMA transfer...\n");
    tma_copy_kernel<<<G.grid(), config::NUM_THREADS, G.dynamic_shared_memory()>>>(G);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Re-initialize device 1 for accurate test
    CUDACHECK(cudaSetDevice(1));
    initKernel<<<gridSize, blockSize>>>(d1_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Timed TMA transfer
    printf("\nStarting timed TMA transfer: Device 0 -> Device 1\n");
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(start));
    
    tma_copy_kernel<<<G.grid(), config::NUM_THREADS, G.dynamic_shared_memory()>>>(G);
    
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaDeviceSynchronize());
    
    float milliseconds = 0;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    double seconds = milliseconds / 1000.0;
    double gigabytes = dataSize / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_GBps = gigabytes / seconds;
    
    printf("\nTMA Transfer Results:\n");
    printf("---------------------\n");
    printf("Transfer time: %.3f ms\n", milliseconds);
    printf("TMA Bandwidth: %.2f GB/s\n", bandwidth_GBps);
    
    // Verify correctness on device 1
    printf("\nVerifying data correctness on Device 1...\n");
    CUDACHECK(cudaSetDevice(1));
    
    int* d_errorCount;
    CUDACHECK(cudaMalloc(&d_errorCount, sizeof(int)));
    CUDACHECK(cudaMemset(d_errorCount, 0, sizeof(int)));
    
    verifyKernel<<<gridSize, blockSize>>>(d1_data, srcValue, numElements, d_errorCount);
    CUDACHECK(cudaDeviceSynchronize());
    
    int h_errorCount;
    CUDACHECK(cudaMemcpy(&h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (h_errorCount == 0) {
        printf("✓ Correctness check PASSED: All values match expected value (%.2f)\n", srcValue);
    } else {
        printf("✗ Correctness check FAILED: %d mismatches found\n", h_errorCount);
    }
    
    // Sample a few values for verification
    bf16 sample_bf[10];
    CUDACHECK(cudaMemcpy(sample_bf, d1_data, sizeof(sample_bf), cudaMemcpyDeviceToHost));
    printf("\nFirst 10 values on Device 1 after transfer: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", __bfloat162float(sample_bf[i]));
    }
    printf("\n");
    
    // Cleanup
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaFree(d0_data));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaFree(d1_data));
    CUDACHECK(cudaFree(d_errorCount));
    
    printf("\nTest completed successfully!\n");
    
    return 0;
}