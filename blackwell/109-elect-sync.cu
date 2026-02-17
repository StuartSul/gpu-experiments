#include "kittens.cuh"

using namespace kittens;

using tile = st_bf<16, 16>;
using layout = gl<bf16, 1, 1, -1, -1, tile>;

__global__ void manual_branching_kernel(const __grid_constant__ layout A) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    auto &smem = al.allocate<tile>();

    // Initialize mbarrier
    __shared__ semaphore arrived;
    if (threadIdx.x == 0) {
        init_semaphore(arrived, 0, 1);
    }
    __syncthreads();

    // Issue TMA load
    if (warpgroup::groupid() == 0) {
        if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
            tma::expect_bytes(arrived, sizeof(smem));
            tma::load_async(smem, A, {0, 0}, arrived);
        }
    }
    wait(arrived, 0);
}

__global__ void elect_sync_kernel(const __grid_constant__ layout A) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    auto &smem = al.allocate<tile>();

    // Initialize mbarrier
    __shared__ semaphore arrived;
    if (threadIdx.x == 0) {
        init_semaphore(arrived, 0, 1);
    }
    __syncthreads();

    // Issue TMA load
    if (warpgroup::groupid() == 0) {
        if (warpgroup::warpid() == 0 && warp::elect_leader()) {
            tma::expect_bytes(arrived, sizeof(smem));
            tma::load_async(smem, A, {0, 0}, arrived);
        }
    }
    wait(arrived, 0);
}

__host__ int main() {
    const int M = 16, N = 16;
    bf16 *d_A;
    CUDACHECK(cudaMalloc(&d_A, M * N * sizeof(bf16)));
    CUDACHECK(cudaMemset(d_A, 1.0, M * N * sizeof(bf16)));

    layout A{d_A, nullptr, nullptr, M, N};

    CUDACHECK(cudaFuncSetAttribute(manual_branching_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY-1024));
    manual_branching_kernel<<<1, 1, MAX_SHARED_MEMORY-1024>>>(A);
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaFuncSetAttribute(elect_sync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY-1024));
    elect_sync_kernel<<<1, 1, MAX_SHARED_MEMORY-1024>>>(A);
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaFree(d_A));
    return 0;
}
