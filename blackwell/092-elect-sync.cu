#include "kittens.cuh"

using namespace kittens;

__global__ void kernel() {
    uint32_t elected = 0;
    asm volatile(
        "{.reg .pred P;\n"
        " elect.sync _|P, %1;\n"
        " selp.u32 %0, 1, 0, P;}\n"
        : "+r"(elected)
        : "r"(0xFFFFFFFF)
    );
    
    if (elected)
        printf("ThreadIdx.x: %d\n", threadIdx.x);    
}

__host__ int main() {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY);
    kernel<<<1, 128, MAX_SHARED_MEMORY>>>();
    CUDACHECK(cudaDeviceSynchronize());
    return 0;
}
