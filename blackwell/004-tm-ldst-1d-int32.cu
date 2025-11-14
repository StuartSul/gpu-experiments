/*
    Tensor memory simple load/store

    Observations:
        - Each warp can only access 32 lanes of the TM
        - Thus, whatever TM address we pass, the lane value is ** AUTOMATICALLY mod-32'ed ** and added warp_id * 32.
        - For instance, if I pass in 0x0060'0000 to warp 0 (which refer to lane 96), it will just mod the lane and access lane 0
        - Likely, if I pass in lane 0 to warp 3, it will access lane 96
*/


#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

// Kernel globals
struct globals {
    gl<int, 1, 1, 1, -1> tensor;

    __host__ inline dim3 grid() { return dim3(1); } // use single block
    __host__ inline dim3 block() { return dim3(128); } // use single warpgroup
    __host__ inline int dynamic_shared_memory() { return MAX_SHARED_MEMORY - 1024; }
};

// Kernel implementation
__global__ void kernel(const __grid_constant__ globals G) {
    // Allocate Tensor Memory (TM) for 1-CTA group 
    __shared__ uint32_t tm_addr_shared;
    uint32_t tm_addr = 0;
    uint32_t n_cols = 512; // full TM allocation
    if (threadIdx.x < 32) { // must be performed by a single warp in the CTA
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;"
            :: "l"((uint64_t)&tm_addr_shared), "r"(n_cols)
        ); // __syncwarp() naturally happens here
        // After relinquish_alloc_permit, it becomes illegal for this CTA to call tcgen05.alloc
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    tm_addr = tm_addr_shared; // Move from shared memory into register

    // TM store launched by threads 0, ..., 127
    int src = threadIdx.x;
    asm volatile("tcgen05.st.sync.aligned.16x64b.x1.b32 [%0], {%1};"
        :: "r"(tm_addr + (3 - (threadIdx.x / 32)) * 0x0020'0000), "r"(src)
    );
    asm volatile("tcgen05.wait::st.sync.aligned;"); // waits for st issued by current thread
    asm volatile("bar.sync %0, %1;" :: "n"(0), "n"(128)); // warpgroup sync

    // TM load launched by threads 0, ..., 127
    int dst = 0; // to truly check if value got loaded
    asm volatile("tcgen05.ld.sync.aligned.16x64b.x1.b32 {%0}, [%1];"
        : "=r"(dst) // although different addr, mod 32 will make TM access per warp the same
        : "r"(tm_addr + (threadIdx.x / 32) * 0x0020'0000)
    );
    asm volatile("tcgen05.wait::st.sync.aligned;"); // waits for st issued by current thread
    asm volatile("bar.sync %0, %1;" :: "n"(0), "n"(128)); // warpgroup sync

    // Save to global memory for validation
    G.tensor.raw_ptr[threadIdx.x] = dst;

    // De-allocate TM for 1-CTA group
    // Without this, CUDA will raise unfreed tensor memory error
    if (threadIdx.x < 32) { // must be performed by a single warp in the CTA
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tm_addr), "r"(n_cols)
        );
    }
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    kittens::py::bind_kernel<kernel>(m, "kernel",
        &globals::tensor
    );
}
