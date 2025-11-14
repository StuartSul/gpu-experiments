/*
    Tensor memory allocation tests

    Observations:
        - tcgen05.alloc will hang if there is no free tensor memory available
        - the returned TM address can be treated like address "0" and can be added (row << 16) | col
        - the addresses returned are "usually" incremental, but sometimes they skip and hop

    Unless in special need, I think it is just best to allocate the entire TM (128x512) at the beginning
    of a kernel and just address accordingly
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

// Kernel globals
struct globals {
    gl<int, 1, -1, -1, -1> tensor;

    __host__ inline dim3 grid() { return dim3(1); } // use single block
    __host__ inline dim3 block() { return dim3(128); } // use single warpgroup
    __host__ inline int dynamic_shared_memory() { return MAX_SHARED_MEMORY - 1024; }
};

// Kernel implementation
__global__ void kernel(const __grid_constant__ globals G) {
    // Allocate Tensor Memory (TM) for 1-CTA group 
    __shared__ uint32_t tm_addr_shared;
    if (threadIdx.x < 32) { // must be performed by a single warp in the CTA
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;"
            :: "l"((uint64_t)&tm_addr_shared), "r"(32)
        ); // __syncwarp() naturally happens here
        printf("First addr: %u\n", tm_addr_shared);
        
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;"
            :: "l"((uint64_t)&tm_addr_shared), "r"(64)
        );
        printf("Second addr: %u\n", tm_addr_shared);
        
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;"
            :: "l"((uint64_t)&tm_addr_shared), "r"(128)
        );
        printf("Third addr: %u\n", tm_addr_shared);
        
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;"
            :: "l"((uint64_t)&tm_addr_shared), "r"(256)
        );
        printf("Fourth addr: %u\n", tm_addr_shared);

        // This will cause cause deadlock
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;"
            :: "l"((uint64_t)&tm_addr_shared), "r"(32)
        );
        printf("Fifth addr: %u\n", tm_addr_shared);

        // This will cause cause deadlock
        // asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;"
        //     :: "l"((uint64_t)&tm_addr_shared), "r"(32)
        // );

        // After relinquish_alloc_permit, it becomes illegal for this CTA to call tcgen05.alloc
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    
    // Since we don't de-allocate TM for 1-CTA group, CUDA error will be raised
    __syncthreads();
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    kittens::py::bind_kernel<kernel>(m, "kernel",
        &globals::tensor
    );
}
