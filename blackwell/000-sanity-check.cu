#include "gpu-experiments.cuh"

__global__ void kernel() {
    printf("Block: %d/%d | Thread: %d/%d\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

int main() {
    kernel<<<148, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
