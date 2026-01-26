/*
    Note that `cudaLaunchAttributeProgrammaticStreamSerialization` attribute alone removes memory flush
    requirement, which speeds up the kernel launches already. Thus, we see speedups even with NUM_BLOCKS = 148.
*/

#include "kittens.cuh"

constexpr int NUM_BLOCKS = 148;
constexpr int BLOCK_SIZE = 512;
constexpr int N = NUM_BLOCKS*BLOCK_SIZE*32;
constexpr int ITERATIONS = 100;

__global__ void kernel_baseline(float* out, const float* in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int idx = tid; idx < n; idx += stride) {
        float val = in[idx];
        val = sinf(val) * cosf(val) + val;
        out[idx] = val;
    }
}

__global__ void kernel_pdl(float* out, const float* in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    cudaGridDependencySynchronize();

    for (int idx = tid; idx < n; idx += stride) {
        float val = in[idx];   
        val = sinf(val) * cosf(val) + val;
        out[idx] = val;
    }

    cudaTriggerProgrammaticLaunchCompletion();
}

int main() {
    float *d_in, *d_out;
    CUDACHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDACHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDACHECK(cudaMemset(d_in, 1, N * sizeof(float)));

    CUDACHECK(cudaFuncSetAttribute(kernel_baseline, cudaFuncAttributeMaxDynamicSharedMemorySize, kittens::MAX_SHARED_MEMORY));
    CUDACHECK(cudaFuncSetAttribute(kernel_pdl, cudaFuncAttributeMaxDynamicSharedMemorySize, kittens::MAX_SHARED_MEMORY));

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(NUM_BLOCKS);
    config.blockDim = dim3(BLOCK_SIZE);
    config.dynamicSmemBytes = kittens::MAX_SHARED_MEMORY;
    config.stream = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    // Warmup
    for (int i = 0; i < 10; i++) {
        kernel_baseline<<<NUM_BLOCKS, BLOCK_SIZE, kittens::MAX_SHARED_MEMORY>>>(d_out, d_in, N);
        CUDACHECK(cudaLaunchKernelEx(&config, kernel_pdl, d_out, d_in, N));
    }
    CUDACHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    // Benchmark baseline
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        kernel_baseline<<<NUM_BLOCKS, BLOCK_SIZE, kittens::MAX_SHARED_MEMORY>>>(d_out, d_in, N);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));
    float baseline_ms;
    CUDACHECK(cudaEventElapsedTime(&baseline_ms, start, stop));

    // Benchmark PDL
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        CUDACHECK(cudaLaunchKernelEx(&config, kernel_pdl, d_out, d_in, N));
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));
    float pdl_ms;
    CUDACHECK(cudaEventElapsedTime(&pdl_ms, start, stop));

    printf("Baseline: %.2f ms (%.2f us/iter)\n", baseline_ms, baseline_ms / ITERATIONS * 1000);
    printf("PDL:      %.2f ms (%.2f us/iter)\n", pdl_ms, pdl_ms / ITERATIONS * 1000);
    printf("Speedup:  %.2fx\n", baseline_ms / pdl_ms);

    CUDACHECK(cudaFree(d_in));
    CUDACHECK(cudaFree(d_out));
    return 0;
}
