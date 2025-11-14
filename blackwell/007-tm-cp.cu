/*
    Observations:
    - warpx4 means it will be multicasted to TM lanes 0-31, 32-63, 64-95, 96-127
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct globals {
    static constexpr int BLOCK_SIZE = 128;

    gl<int, 1, 1, BLOCK_SIZE, BLOCK_SIZE> tensor;

    __host__ inline dim3 grid() { return dim3(1); } // use single block
    __host__ inline dim3 block() { return dim3(BLOCK_SIZE); } // use single warpgroup
    __host__ inline int dynamic_shared_memory() { return MAX_SHARED_MEMORY - 1024; }
};

__global__ void kernel(const __grid_constant__ globals G) {
    // Prefix. Comments are in the earlier tests
    __shared__ uint32_t tm_addr_shared;
    uint32_t tm_addr = 0;
    uint32_t n_cols = 512;
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;"
            :: "l"((uint64_t)&tm_addr_shared), "r"(n_cols)
        );
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    tm_addr = tm_addr_shared;
    // Prefix done
    // --------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------

    // Initialize an mbarrier
    __shared__ uint64_t mbarrier;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "l"(__cvta_generic_to_shared(&mbarrier)), "r"(1));
    }
    __syncthreads();

    // Fill in 128 x 128 shared memory tile
    extern __shared__ int32_t smem[];
    int32_t *st = reinterpret_cast<int32_t *>(
        (reinterpret_cast<uint64_t>(&smem[0]) + 1023) / 1024 * 1024
    );
    if (threadIdx.x == 0) {
        for (int i = 0; i < globals::BLOCK_SIZE; ++i) {
            for (int j = 0; j < globals::BLOCK_SIZE; ++j) {
                st[i * globals::BLOCK_SIZE + j] = i * 10000 + j;
            }
        }
    }
    __syncthreads();

    // Generated shared memory descriptor
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
    uint64_t st_addr = reinterpret_cast<uint64_t>(&st[0]);
    uint64_t s_desc = 
        (((st_addr & 0x3FFFFULL) >> 4) << 0) | // bits 00-13: smem addr
        (0ULL << 14)                         | // bits 14-15: SBZ
        (((128ULL & 0x3FFFFULL) >> 4) << 16) | // bits 16-29: leading dimension relative byte offset
        (0ULL << 30)                         | // bits 30-31: SBZ
        (((128ULL & 0x3FFFFULL) >> 4) << 32) | // bits 32-45: stride dimension byte offset (16B atom per row x 8)
        (1ULL << 46)                         | // bits 46-48: fixed constant of 1
        (0ULL << 49)                         | // bits 49-51: matrix byte offset (0 if 1024B-aligned)
        (0ULL << 52)                         | // bits 52-52: leading dimension stride mode (0 for relative)
        (0ULL << 53)                         | // bits 53-60: fixed constant of 0
        (0ULL << 61);                          // bits 61-63: swizzling mode used (0 for no swizzling)

    // Perform the async copy from smem to tmem
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" // warpx4 means it will be multicasted to TM lanes 0-31, 32-63, 64-95, 96-127
            :: "r"(tm_addr), "l"(s_desc)
        );
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "l"(__cvta_generic_to_shared(&mbarrier))
        );
        asm volatile("tcgen05.fence::before_thread_sync;"); // adhering to tcgen05 docs
    }

    // Wait for the operation to complete
    asm volatile(
        "{.reg .pred P1;                                            \t\n"
        "BAR_WAIT:                                                  \t\n"
        "    mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1; \t\n"
        "    @P1 bra.uni DONE;                                      \t\n"
        "    bra.uni BAR_WAIT;                                      \t\n"
        "DONE:                                                      \t\n}"
        :: "l"(__cvta_generic_to_shared(&mbarrier)), "n"(0)
    );
    asm volatile("tcgen05.fence::after_thread_sync;"); // fence is needed before tcgen05.ld

    // --------------------------------------------------------------------------------
    // --------------------------------------------------------------------------------
    // Suffix. Comments are in the earlier tests
    int dst[globals::BLOCK_SIZE];
    for (int i = 0; i < globals::BLOCK_SIZE; i++)
        dst[i] = 0;
    for (int i = 0; i < globals::BLOCK_SIZE; i++)
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
            : "=r"(dst[i])
            : "r"(tm_addr + i)
        );
    asm volatile("tcgen05.wait::st.sync.aligned;");
    asm volatile("bar.sync %0, %1;" :: "n"(0), "n"(globals::BLOCK_SIZE));
    for (int i = 0; i < globals::BLOCK_SIZE; i++)
        G.tensor.raw_ptr[threadIdx.x * globals::BLOCK_SIZE + i] = dst[i];
    if (threadIdx.x < 32)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tm_addr), "r"(n_cols)
        );
    // Suffix done
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    kittens::py::bind_kernel<kernel>(m, "kernel",
        &globals::tensor
    );
}
