/*

Observation: 
1. mbarrier init is simply st.shared with a specific value
2. thus it's very parallelizable

*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define NUM_BARRIERS 12

__global__ void kernel_parallel(uint64_t *cycles_out) {
    __shared__ uint64_t barriers[NUM_BARRIERS];

    int tid = threadIdx.x;
    uint32_t ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barriers[tid]));

    __syncwarp();
    uint64_t start = clock64();

    if (tid < NUM_BARRIERS) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(ptr), "r"(1));
    }

    __syncwarp();
    uint64_t end = clock64();

    if (tid == 0) *cycles_out = end - start;
}

__global__ void kernel_sequential(uint64_t *cycles_out) {
    __shared__ uint64_t barriers[NUM_BARRIERS];

    int tid = threadIdx.x;

    __syncwarp();
    uint64_t start = clock64();

    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < NUM_BARRIERS; i++) {
            uint32_t p = static_cast<uint32_t>(__cvta_generic_to_shared(&barriers[i]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(p), "r"(1));
        }
    }

    __syncwarp();
    uint64_t end = clock64();

    if (tid == 0) *cycles_out = end - start;
}

int main() {
    uint64_t *d_cycles;
    uint64_t h_cycles;
    cudaMalloc(&d_cycles, sizeof(uint64_t));

    kernel_parallel<<<1, 32>>>(d_cycles);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("PARALLEL   (12 threads, 1 init each): %lu cycles\n", h_cycles);

    kernel_sequential<<<1, 32>>>(d_cycles);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_cycles, d_cycles, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("SEQUENTIAL (1 thread, 12 inits):      %lu cycles\n", h_cycles);

    cudaFree(d_cycles);

    return 0;
}
