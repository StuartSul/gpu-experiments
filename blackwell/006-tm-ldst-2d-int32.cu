/*
    Tensor memory 2D load/store

    Observations:
        - For 16xNb ld/st ops, the column address must be aligned to the column size of the shape
        - For instance, the given column must be a multiple of 2 for 16x64b (64b = 2 TM columns)
        - **** CUDA does NOT raise error for out-of-bounds TM addressing ****

    Now we have a way to reliably check the contents of TM. This is good.
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

// Kernel globals
struct globals {
    static constexpr int BLOCK_SIZE = 128;

    gl<int, 1, 1, BLOCK_SIZE, BLOCK_SIZE> tensor;

    __host__ inline dim3 grid() { return dim3(1); } // use single block
    __host__ inline dim3 block() { return dim3(BLOCK_SIZE); } // use single warpgroup
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

    // Assign registers (128x128 values, divided by 128 WG threads)
    // Each thread contains an entire row
    int src[globals::BLOCK_SIZE];
    int dst[globals::BLOCK_SIZE];
    for (int i = 0; i < globals::BLOCK_SIZE; i++) {
        src[i] = (threadIdx.x * 10000) + i;
        dst[i] = 9999'9999; // for validation
    }
    
    // TM store launched by threads 0, ..., 127
    for (int i = 0; i < globals::BLOCK_SIZE; i++) {
        // Each iteration handles an entire single column of TM
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
            :: "r"(tm_addr + i /* lane 0, column i */), "r"(src[i])
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;"); // waits for st issued by current thread
    asm volatile("bar.sync %0, %1;" :: "n"(0), "n"(globals::BLOCK_SIZE)); // warpgroup sync

    // TM load launched by threads 0, ..., 127
    for (int i = 0; i < globals::BLOCK_SIZE; i++) {
        // Each iteration handles an entire single column of TM
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
            : "=r"(dst[i])
            : "r"(tm_addr + i /* lane 0, column i */)
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;"); // waits for st issued by current thread
    asm volatile("bar.sync %0, %1;" :: "n"(0), "n"(globals::BLOCK_SIZE)); // warpgroup sync

    // Save to global memory for validation
    for (int i = 0; i < globals::BLOCK_SIZE; i++) {
        // *this is really bad*
        G.tensor.raw_ptr[threadIdx.x * globals::BLOCK_SIZE + i] = dst[i];
    }

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
