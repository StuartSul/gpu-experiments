/*
    Same-node NVLink unidirectional bandwidth via the copy engine (cudaMemcpyPeerAsync).
    Device srcDev -> Device dstDev, all GB300 on the same NVLink fabric.

    GPU0 -> GPUN copy-engine ablation (10 GB):
        0 -> 1: 726.46 GB/s
        0 -> 2: 726.35 GB/s
        0 -> 3: 726.52 GB/s
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
    
    printf("NVLink Unidirectional Bandwidth Test (Copy Engine)\n");
    printf("==================================================\n");
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
    // Limit grid size to avoid excessive blocks
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
    
    // Create events on the source device for timing
    CUDACHECK(cudaSetDevice(srcDev));
    cudaStream_t stream;
    cudaEvent_t start, stop;
    CUDACHECK(cudaStreamCreate(&stream));
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    
    // Enable peer access from source to destination (ignore if already enabled)
    cudaError_t peerStatus = cudaDeviceEnablePeerAccess(dstDev, 0);
    if (peerStatus != cudaSuccess && peerStatus != cudaErrorPeerAccessAlreadyEnabled) {
        CUDACHECK(peerStatus);
    }
    cudaGetLastError(); // clear the sticky error if access was already enabled
    
    // Warm up run
    printf("\nPerforming warm-up transfer...\n");
    CUDACHECK(cudaMemcpyPeerAsync(dst_data, dstDev, src_data, srcDev, dataSize, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    
    // Timed transfer: source -> destination
    printf("\nStarting timed transfer: Device %d -> Device %d\n", srcDev, dstDev);
    CUDACHECK(cudaEventRecord(start, stream));
    CUDACHECK(cudaMemcpyPeerAsync(dst_data, dstDev, src_data, srcDev, dataSize, stream));
    CUDACHECK(cudaEventRecord(stop, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    
    // Calculate elapsed time and bandwidth
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
    CUDACHECK(cudaStreamDestroy(stream));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));
    
    CUDACHECK(cudaSetDevice(dstDev));
    CUDACHECK(cudaFree(dst_data));
    CUDACHECK(cudaFree(d_errorCount));
    
    printf("\n");
    return bandwidth_GBps;
}

int main() {
    // Ablation: copy-engine bandwidth from GPU 0 to each peer GPU.
    double bw01 = benchmark(0, 1);
    double bw02 = benchmark(0, 2);
    double bw03 = benchmark(0, 3);
    
    printf("==================== Ablation Summary ====================\n");
    printf("GPU 0 -> GPU 1: %.2f GB/s\n", bw01);
    printf("GPU 0 -> GPU 2: %.2f GB/s\n", bw02);
    printf("GPU 0 -> GPU 3: %.2f GB/s\n", bw03);
    
    return 0;
}
