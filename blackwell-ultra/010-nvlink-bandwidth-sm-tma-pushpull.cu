/*
    Same-node NVLink bandwidth via TMA (tma::load_async -> smem -> tma::store_async).
    "Push+Pull" combo: the 10 GB transfer src -> dst is split in half along the row dimension and
    driven by TWO concurrent kernels:
        - first half  PUSHED by srcDev  (kernel on src: local read  -> NVLink write to dst)
        - second half PULLED by dstDev  (kernel on dst: NVLink read from src -> local write)

    GPU 0 <-> GPU N push+pull ablation (10 GB aggregate):
        0 push / 1 pull: 717.67 GB/s
        0 push / 2 pull: 717.74 GB/s
        0 push / 3 pull: 717.67 GB/s
*/

#include "kittens.cuh"
#include <chrono>

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
    // Grid covering exactly half of the row-blocks (used for each of the two concurrent kernels).
    __host__ inline dim3 half_grid() const {
        return dim3(tensor.cols() / COL_BLOCK_SIZE, (tensor.rows() / ROW_BLOCK_SIZE) / 2, tensor.depth());
    }
    __host__ inline int row_block_split() const {
        return (tensor.rows() / ROW_BLOCK_SIZE) / 2;
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

// row_block_offset lets a launch operate on a sub-range of row-blocks so the two concurrent
// kernels (push half + pull half) cover disjoint regions of the tensor.
__launch_bounds__(config::NUM_THREADS, 6)
__global__ void tma_copy_kernel(const __grid_constant__ globals G, int row_block_offset) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::tile &tile = allocator.allocate<globals::tile>();

    const int depth_idx = blockIdx.z;
    const int row_block_idx = blockIdx.y + row_block_offset;
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

double benchmark(int srcDev, int dstDev) {
    // Configuration
    const size_t dataSize = 10ULL * 1024 * 1024 * 1024;  // 10 GB
    const size_t numElements = dataSize / sizeof(bf16);
    const float srcValue = 3.14f;
    const float dstInitValue = 0.0f;
    
    printf("NVLink Bandwidth Test (TMA Push+Pull Combo)\n");
    printf("============================================\n");
    printf("Data size: %.2f GB (split 50/50)\n", dataSize / (1024.0 * 1024.0 * 1024.0));
    printf("Number of bf16 elements: %zu\n", numElements);
    printf("Using ThunderKittens TMA primitives\n");
    printf("First half PUSHED by Device %d, second half PULLED by Device %d\n\n", srcDev, dstDev);
    
    // Enable peer access in both directions (ignore if already enabled)
    CUDACHECK(cudaSetDevice(srcDev));
    cudaError_t peerStatus = cudaDeviceEnablePeerAccess(dstDev, 0);
    if (peerStatus != cudaSuccess && peerStatus != cudaErrorPeerAccessAlreadyEnabled) {
        CUDACHECK(peerStatus);
    }
    CUDACHECK(cudaSetDevice(dstDev));
    peerStatus = cudaDeviceEnablePeerAccess(srcDev, 0);
    if (peerStatus != cudaSuccess && peerStatus != cudaErrorPeerAccessAlreadyEnabled) {
        CUDACHECK(peerStatus);
    }
    cudaGetLastError(); // clear the sticky error if access was already enabled
    printf("Peer access enabled between Device %d and Device %d\n\n", srcDev, dstDev);

    // Allocate source memory on the source device
    bf16* src_data;
    CUDACHECK(cudaSetDevice(srcDev));
    CUDACHECK(cudaMalloc(&src_data, dataSize));
    printf("Allocated %.2f GB on Device %d\n", dataSize / (1024.0 * 1024.0 * 1024.0), srcDev);

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
    printf("Allocated %.2f GB on Device %d\n", dataSize / (1024.0 * 1024.0 * 1024.0), dstDev);
    
    // Initialize destination memory with 0
    initKernel<<<gridSize, blockSize>>>(dst_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    printf("Initialized Device %d memory with value: %.2f\n\n", dstDev, dstInitValue);
    
    // Create globals (tensor[0] = source on srcDev, tensor[1] = destination on dstDev).
    // The same globals is used by both launches; the kernel body is identical, only the launching
    // device differs (srcDev pushes, dstDev pulls) and the row_block_offset selects the half.
    bf16 *data[2] = {src_data, dst_data};
    globals G {
        .tensor = globals::parallel_layout{data, nullptr, 5, 32768, 32768},
        .dev_idx = srcDev
    };
    const int split = G.row_block_split();
    const dim3 half = G.half_grid();
    const int smem = G.dynamic_shared_memory();

    auto launch_both = [&]() {
        // Push first half from srcDev (row-blocks [0, split))
        CUDACHECK(cudaSetDevice(srcDev));
        tma_copy_kernel<<<half, config::NUM_THREADS, smem>>>(G, 0);
        // Pull second half from dstDev (row-blocks [split, 2*split))
        CUDACHECK(cudaSetDevice(dstDev));
        tma_copy_kernel<<<half, config::NUM_THREADS, smem>>>(G, split);
    };

    // Warm up run
    printf("Performing warm-up push+pull transfer...\n");
    launch_both();
    CUDACHECK(cudaSetDevice(srcDev));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaSetDevice(dstDev));
    CUDACHECK(cudaDeviceSynchronize());
    
    // Re-initialize the destination for an accurate test
    CUDACHECK(cudaSetDevice(dstDev));
    initKernel<<<gridSize, blockSize>>>(dst_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Timed transfer: launch both kernels concurrently, then wait for both devices.
    // Two GPUs are involved, so we time the aggregate with a host-side wall clock.
    printf("\nStarting timed push+pull transfer (Device %d push + Device %d pull)\n", srcDev, dstDev);
    auto t0 = std::chrono::high_resolution_clock::now();
    launch_both();
    CUDACHECK(cudaSetDevice(srcDev));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaSetDevice(dstDev));
    CUDACHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    double milliseconds = seconds * 1000.0;
    double gigabytes = dataSize / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_GBps = gigabytes / seconds;
    
    printf("\nTMA Transfer Results:\n");
    printf("---------------------\n");
    printf("Transfer time: %.3f ms\n", milliseconds);
    printf("Aggregate TMA Bandwidth: %.2f GB/s\n", bandwidth_GBps);
    
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

    CUDACHECK(cudaSetDevice(dstDev));
    CUDACHECK(cudaFree(dst_data));
    CUDACHECK(cudaFree(d_errorCount));
    
    printf("\n");
    return bandwidth_GBps;
}

int main() {
    // Ablation: GPU 0 pushes the first half while peer GPU pulls the second half.
    double bw01 = benchmark(0, 1);
    double bw02 = benchmark(0, 2);
    double bw03 = benchmark(0, 3);
    
    printf("==================== Ablation Summary ====================\n");
    printf("GPU 0 push / GPU 1 pull: %.2f GB/s\n", bw01);
    printf("GPU 0 push / GPU 2 pull: %.2f GB/s\n", bw02);
    printf("GPU 0 push / GPU 3 pull: %.2f GB/s\n", bw03);
    
    return 0;
}
