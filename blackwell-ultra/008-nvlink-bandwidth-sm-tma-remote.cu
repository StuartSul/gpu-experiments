/*
    Launched from GPU 0; movement from GPU 1 -> GPU 2 (10 GB): 591.73 GB/s.

    Observations
    - Only 12% drop compared to 001-nvlink-bandwidth-sm-tma (669 GB/s) despite two NVLink hops
    - Because NVLink is bidirectional, GPU 0's read and write ports can overlap!!
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

// Enable peer access from `fromDev` to `toDev` (ignore if already enabled)
static void enablePeerAccess(int fromDev, int toDev) {
    if (fromDev == toDev) return;
    CUDACHECK(cudaSetDevice(fromDev));
    cudaError_t peerStatus = cudaDeviceEnablePeerAccess(toDev, 0);
    if (peerStatus != cudaSuccess && peerStatus != cudaErrorPeerAccessAlreadyEnabled) {
        CUDACHECK(peerStatus);
    }
    cudaGetLastError(); // clear the sticky error if access was already enabled
}

double benchmark(int launchDev, int srcDev, int dstDev) {
    // Configuration
    const size_t dataSize = 10ULL * 1024 * 1024 * 1024;  // 10 GB
    const size_t numElements = dataSize / sizeof(bf16);
    const float srcValue = 3.14f;
    const float dstInitValue = 0.0f;
    
    printf("Remote NVLink Bandwidth Test (TMA-based Copy)\n");
    printf("==============================================\n");
    printf("Data size: %.2f GB\n", dataSize / (1024.0 * 1024.0 * 1024.0));
    printf("Number of bf16 elements: %zu\n", numElements);
    printf("Using ThunderKittens TMA primitives\n");
    printf("Driver: Device %d   Direction: Device %d -> Device %d\n\n", launchDev, srcDev, dstDev);
    
    // The driver device's SMs must reach both the source and destination memory, and the
    // destination must be reachable from the source path as well. Enable peer access for every
    // ordered pair among the three devices we touch.
    enablePeerAccess(launchDev, srcDev);
    enablePeerAccess(launchDev, dstDev);
    enablePeerAccess(srcDev, launchDev);
    enablePeerAccess(srcDev, dstDev);
    enablePeerAccess(dstDev, launchDev);
    enablePeerAccess(dstDev, srcDev);
    printf("Peer access enabled among Devices %d, %d, %d\n\n", launchDev, srcDev, dstDev);

    // Allocate source memory on the source device
    bf16* src_data;
    CUDACHECK(cudaSetDevice(srcDev));
    CUDACHECK(cudaMalloc(&src_data, dataSize));
    printf("Allocated %.2f GB on Device %d (source)\n", dataSize / (1024.0 * 1024.0 * 1024.0), srcDev);

    // Initialize source memory with 3.14
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    gridSize = min(gridSize, 65536);

    initKernel<<<gridSize, blockSize>>>(src_data, srcValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    printf("Initialized Device %d memory with value: %.2f\n", srcDev, srcValue);
    
    // Allocate destination memory on the destination device
    bf16* dst_data;
    CUDACHECK(cudaSetDevice(dstDev));
    CUDACHECK(cudaMalloc(&dst_data, dataSize));
    printf("Allocated %.2f GB on Device %d (destination)\n", dataSize / (1024.0 * 1024.0 * 1024.0), dstDev);
    
    // Initialize destination memory with 0
    initKernel<<<gridSize, blockSize>>>(dst_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    printf("Initialized Device %d memory with value: %.2f\n\n", dstDev, dstInitValue);
    
    // Create events for timing on the driver device (where the copy kernel runs)
    CUDACHECK(cudaSetDevice(launchDev));
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    
    // Create globals (tensor[0] = source on srcDev, tensor[1] = destination on dstDev)
    bf16 *data[2] = {src_data, dst_data};
    globals G {
        .tensor = globals::parallel_layout{data, nullptr, 5, 32768, 32768},
        .dev_idx = launchDev
    };

    // Warm up run (kernel launches on the driver device)
    printf("Performing warm-up TMA transfer (driver Device %d)...\n", launchDev);
    CUDACHECK(cudaSetDevice(launchDev));
    tma_copy_kernel<<<G.grid(), config::NUM_THREADS, G.dynamic_shared_memory()>>>(G);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Re-initialize the destination for an accurate test
    CUDACHECK(cudaSetDevice(dstDev));
    initKernel<<<gridSize, blockSize>>>(dst_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Timed TMA transfer
    printf("\nStarting timed TMA transfer: Device %d -> Device %d (driver Device %d)\n", srcDev, dstDev, launchDev);
    CUDACHECK(cudaSetDevice(launchDev));
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
    
    // Verify correctness on the destination device
    printf("\nVerifying data correctness on Device %d...\n", dstDev);
    CUDACHECK(cudaSetDevice(dstDev));
    
    int* d_errorCount;
    CUDACHECK(cudaMalloc(&d_errorCount, sizeof(int)));
    CUDACHECK(cudaMemset(d_errorCount, 0, sizeof(int)));
    
    verifyKernel<<<gridSize, blockSize>>>(dst_data, srcValue, numElements, d_errorCount);
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
    CUDACHECK(cudaMemcpy(sample_bf, dst_data, sizeof(sample_bf), cudaMemcpyDeviceToHost));
    printf("\nFirst 10 values on Device %d after transfer: ", dstDev);
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", __bfloat162float(sample_bf[i]));
    }
    printf("\n");
    
    // Cleanup
    CUDACHECK(cudaSetDevice(srcDev));
    CUDACHECK(cudaFree(src_data));

    CUDACHECK(cudaSetDevice(launchDev));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    
    CUDACHECK(cudaSetDevice(dstDev));
    CUDACHECK(cudaFree(dst_data));
    CUDACHECK(cudaFree(d_errorCount));
    
    printf("\n");
    return bandwidth_GBps;
}

int main() {
    // Driver GPU 0 copies data from GPU 1 to GPU 2.
    double bw = benchmark(/*launchDev=*/0, /*srcDev=*/1, /*dstDev=*/2);
    
    printf("==================== Summary ====================\n");
    printf("Driver GPU 0, GPU 1 -> GPU 2: %.2f GB/s\n", bw);
    
    return 0;
}
