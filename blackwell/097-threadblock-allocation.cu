#include "kittens.cuh"

using namespace kittens;

__global__ void kernel_unclustered() {
    int smid;
    asm volatile("{mov.u32 %0, %smid;}" : "=r"(smid));
    int nsmid;
    asm volatile("{mov.u32 %0, %nsmid;}" : "=r"(nsmid));

    printf("Block ID: %3u | SM ID: %3d / %3d | Cluster ID: %2d / %2d\n", blockIdx.x, smid, nsmid, blockIdx.x, gridDim.x);
}

__cluster_dims__(2) __global__ void kernel_2clustered() {
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

__cluster_dims__(4) __global__ void kernel_4clustered() {
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

__cluster_dims__(8) __global__ void kernel_8clustered() {
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

__cluster_dims__(16) __global__ void kernel_16clustered() {
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
    cudaFuncSetAttribute(kernel_unclustered, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY);
    kernel_unclustered<<<148, 1, MAX_SHARED_MEMORY>>>();
    CUDACHECK(cudaDeviceSynchronize());
    printf("=======================================================\n");
    cudaFuncSetAttribute(kernel_2clustered, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY);
    kernel_2clustered<<<148, 1, MAX_SHARED_MEMORY>>>();
    CUDACHECK(cudaDeviceSynchronize());
    printf("=======================================================\n");
    cudaFuncSetAttribute(kernel_4clustered, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY);
    kernel_4clustered<<<148, 1, MAX_SHARED_MEMORY>>>();
    CUDACHECK(cudaDeviceSynchronize());
    printf("=======================================================\n");
    cudaFuncSetAttribute(kernel_8clustered, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY);
    kernel_8clustered<<<144, 1, MAX_SHARED_MEMORY>>>();
    CUDACHECK(cudaDeviceSynchronize());
    printf("=======================================================\n");
    cudaFuncSetAttribute(kernel_16clustered, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY);
    cudaFuncSetAttribute(kernel_16clustered, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    kernel_16clustered<<<144, 1, MAX_SHARED_MEMORY>>>();
    CUDACHECK(cudaDeviceSynchronize());
    return 0;
}
