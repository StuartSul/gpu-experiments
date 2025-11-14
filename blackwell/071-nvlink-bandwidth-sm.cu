/*
    Transfer time: 18.473 ms
    Bandwidth: 541.34 GB/s
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

int main() {
    // Configuration
    const size_t dataSize = 10ULL * 1024 * 1024 * 1024;  // 10 GB
    const size_t numElements = dataSize / sizeof(float);
    const float srcValue = 3.14f;
    const float dstInitValue = 0.0f;
    
    printf("NVLink Bandwidth Test (Kernel-based Copy)\n");
    printf("==========================================\n");
    printf("Data size: %.2f GB\n", dataSize / (1024.0 * 1024.0 * 1024.0));
    printf("Number of float elements: %zu\n", numElements);
    
    // Allocate memory on device 0
    float* d0_data;
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
    float* d1_data;
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMalloc(&d1_data, dataSize));
    printf("Allocated %.2f GB on Device 1\n", dataSize / (1024.0 * 1024.0 * 1024.0));
    
    // Initialize device 1 memory with 0
    initKernel<<<gridSize, blockSize>>>(d1_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    printf("Initialized Device 1 memory with value: %.2f\n\n", dstInitValue);
    
    // Enable peer access
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));
    
    // Create events for timing
    CUDACHECK(cudaSetDevice(0));
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    
    // Warm up run
    printf("Performing warm-up transfer...\n");
    copyKernel<<<gridSize, blockSize>>>(d1_data, d0_data, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Re-initialize device 1 for accurate test
    CUDACHECK(cudaSetDevice(1));
    initKernel<<<gridSize, blockSize>>>(d1_data, dstInitValue, numElements);
    CUDACHECK(cudaDeviceSynchronize());
    
    // Timed kernel copy
    printf("\nStarting timed kernel transfer: Device 0 -> Device 1\n");
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaEventRecord(start));
    
    copyKernel<<<gridSize, blockSize>>>(d1_data, d0_data, numElements);
    
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
    float sample[10];
    CUDACHECK(cudaMemcpy(sample, d1_data, sizeof(sample), cudaMemcpyDeviceToHost));
    printf("\nFirst 10 values on Device 1 after transfer: ");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", sample[i]);
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