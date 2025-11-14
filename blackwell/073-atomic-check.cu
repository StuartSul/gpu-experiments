#include <cuda_runtime.h>
#include <cstdio>

__global__ void atomicAndVolatileKernel(int* sign, int* data) {
    if (*(volatile int*)sign == 0)
        atomicAdd(data, 1);
}

int main() {
    int *d_sign, *d_data;
    
    cudaMalloc(&d_sign, sizeof(int));
    cudaMalloc(&d_data, sizeof(int));
    
    cudaMemset(d_sign, 0, sizeof(int));
    cudaMemset(d_data, 0, sizeof(int));
    
    atomicAndVolatileKernel<<<1, 1>>>(d_sign, d_data);
    cudaDeviceSynchronize();

    int h_data;
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Data value: %d\n", h_data);

    cudaFree(d_sign);
    cudaFree(d_data);
    
    return 0;
}
