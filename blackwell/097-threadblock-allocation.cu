#include "kittens.cuh"

using namespace kittens;

struct globals {
    __host__ dim3 grid() { return dim3(148); }
    __host__ dim3 block() { return dim3(1); }
    __host__ int dynamic_shared_memory() { return MAX_SHARED_MEMORY - 1024; }
};

__global__ void kernel_unclustered(const __grid_constant__ globals g) {
    int smid;
    asm volatile("{mov.u32 %0, %smid;}" : "=r"(smid));
    int nsmid;
    asm volatile("{mov.u32 %0, %nsmid;}" : "=r"(nsmid));

    printf("Block ID: %3u | SM ID: %3d / %3d\n", blockIdx.x, smid, nsmid);
}

__cluster_dims__(2) __global__ void kernel_2clustered(const __grid_constant__ globals g) {
    int smid;
    asm volatile("{mov.u32 %0, %smid;}" : "=r"(smid));
    int nsmid;
    asm volatile("{mov.u32 %0, %nsmid;}" : "=r"(nsmid));
    int clusterid;
    asm volatile("{mov.u32 %0, %clusterid.x;}" : "=r"(clusterid));
    int nclusterid;
    asm volatile("{mov.u32 %0, %nclusterid.x;}" : "=r"(nclusterid));

    printf("Block ID: %3u | SM ID: %3d / %3d | Cluster ID: %2d / %2d\n", blockIdx.x, smid, nsmid, clusterid, nclusterid);
}

__cluster_dims__(4) __global__ void kernel_4clustered(const __grid_constant__ globals g) {
    int smid;
    asm volatile("{mov.u32 %0, %smid;}" : "=r"(smid));
    int nsmid;
    asm volatile("{mov.u32 %0, %nsmid;}" : "=r"(nsmid));
    int clusterid;
    asm volatile("{mov.u32 %0, %clusterid.x;}" : "=r"(clusterid));
    int nclusterid;
    asm volatile("{mov.u32 %0, %nclusterid.x;}" : "=r"(nclusterid));

    printf("Block ID: %3u | SM ID: %3d / %3d | Cluster ID: %2d / %2d\n", blockIdx.x, smid, nsmid, clusterid, nclusterid);
}

int main() {
    globals g;
    cudaFuncSetAttribute(kernel_unclustered, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());
    kernel_unclustered<<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
    CUDACHECK(cudaDeviceSynchronize());
    printf("=======================================================\n");
    cudaFuncSetAttribute(kernel_2clustered, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());
    kernel_2clustered<<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
    CUDACHECK(cudaDeviceSynchronize());
    printf("=======================================================\n");
    cudaFuncSetAttribute(kernel_4clustered, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());
    kernel_4clustered<<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
    CUDACHECK(cudaDeviceSynchronize());
    return 0;
}
