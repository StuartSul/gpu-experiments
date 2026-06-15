/*
    Same-node NVLink bandwidth via SM cores with register-level load/store (dst[i] = src[i]).
    Device srcDev -> Device dstDev, all GB300; threads issue plain LDG/STG over peer memory.

    GPU0 -> GPUN SM-copy ablation (10 GB):
        0 -> 1: 547.61 GB/s
        0 -> 2: 553.98 GB/s
        0 -> 3: 549.10 GB/s
*/

#include "kittens.cuh"

// Kernel to initialize memory with a value
__global__ void initKernel(float* data, float value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        data[i] = value;
    }
}

// Kernel to copy data from one GPU to another with coalesced access
__global__ void copyKernel(float* dst, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Coalesced access pattern: consecutive threads access consecutive memory locations
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
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

double benchmark(int srcDev, int dstDev) {
    // Configuration
    const size_t dataSize = 10ULL * 1024 * 1024 * 1024;  // 10 GB
    const size_t numElements = dataSize / sizeof(float);
    const float srcValue = 3.14f;
    const float dstInitValue = 0.0f;
    
    printf("NVLink Bandwidth Test (Kernel-based Copy)\n");
    printf("==========================================\n");
    printf("Data size: %.2f GB\n", dataSize / (1024.0 * 1024.0 * 1024.0));
    printf("Number of float elements: %zu\n", numElements);
    printf("Direction: Device %d -> Device %d\n", srcDev, dstDev);
    
    // Allocate memory on the source device
    float* src_data;
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
    
    // Allocate memory on the destination device
    float* dst_data;
    CUDACHECK(cudaSetDevice(dstDev));
    CUDACHECK(cudaMalloc(&dst_data, dataSize));
    printf("Allocated %.2f GB on Device %d\n", dataSize / (1024.0 * 1024.0 * 1024.0), dstDev);
    
    // Initialize destination memory with 0
    initKernel<<<gridSize, blockSize>>>(dst_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    printf("Initialized Device %d memory with value: %.2f\n\n", dstDev, dstInitValue);
    
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
    
    // Create events for timing on the source device
    CUDACHECK(cudaSetDevice(srcDev));
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    
    // Warm up run
    printf("\nPerforming warm-up transfer...\n");
    copyKernel<<<gridSize, blockSize>>>(dst_data, src_data, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Re-initialize the destination for an accurate test
    CUDACHECK(cudaSetDevice(dstDev));
    initKernel<<<gridSize, blockSize>>>(dst_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Timed kernel copy
    printf("\nStarting timed kernel transfer: Device %d -> Device %d\n", srcDev, dstDev);
    CUDACHECK(cudaSetDevice(srcDev));
    CUDACHECK(cudaEventRecord(start));
    copyKernel<<<gridSize, blockSize>>>(dst_data, src_data, numElements);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaDeviceSynchronize());
    
    float milliseconds = 0;
    CUDACHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double seconds = milliseconds / 1000.0;
    double gigabytes = dataSize / (1024.0 * 1024.0 * 1024.0);
    double bandwidth_GBps = gigabytes / seconds;
    
    printf("\nTransfer Results:\n");
    printf("-----------------\n");
    printf("Transfer time: %.3f ms\n", milliseconds);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_GBps);
    
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
    float sample[10];
    CUDACHECK(cudaMemcpy(sample, dst_data, sizeof(sample), cudaMemcpyDeviceToHost));
    printf("\nFirst 10 values on Device %d after transfer: ", dstDev);
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", sample[i]);
    }
    printf("\n");
    
    // Cleanup
    CUDACHECK(cudaSetDevice(srcDev));
    CUDACHECK(cudaFree(src_data));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    
    CUDACHECK(cudaSetDevice(dstDev));
    CUDACHECK(cudaFree(dst_data));
    CUDACHECK(cudaFree(d_errorCount));
    
    printf("\n");
    return bandwidth_GBps;
}

int main() {
    // Ablation: SM register-copy bandwidth from GPU 0 to each peer GPU.
    double bw01 = benchmark(0, 1);
    double bw02 = benchmark(0, 2);
    double bw03 = benchmark(0, 3);
    
    printf("==================== Ablation Summary ====================\n");
    printf("GPU 0 -> GPU 1: %.2f GB/s\n", bw01);
    printf("GPU 0 -> GPU 2: %.2f GB/s\n", bw02);
    printf("GPU 0 -> GPU 3: %.2f GB/s\n", bw03);
    
    return 0;
}
