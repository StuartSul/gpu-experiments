#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

// ThunderKittens macro check
#if !defined(KITTENS_HOPPER) || !defined(KITTENS_BLACKWELL)
    #error "KITTENS_HOPPER and KITTENS_BLACKWELL macros must be defined for Blackwell compilation"
#endif

// Kernel configuration
struct config {
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;
};

// Kernel globals
struct globals {
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 128;
    static constexpr int REDUCTION_BLOCK = 128;

    using A_fp8_tile = st_fp8e4m3<ROW_BLOCK, REDUCTION_BLOCK>;
    using B_fp8_tile = st_fp8e4m3<COL_BLOCK, REDUCTION_BLOCK>;
    using C_tile = st_fl<ROW_BLOCK, COL_BLOCK>;

    gl<fp8e4m3, 1, 1, -1, -1, A_fp8_tile> A_fp8; // M x K
    gl<fp8e4m3, 1, 1, -1, -1, B_fp8_tile> B_fp8; // N x K
    gl<float, 1, 1, -1, -1, C_tile> C;           // M x N

    __host__ inline dim3 grid() { return dim3(1); } // use single block
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

// Kernel implementation
__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Warpgroup configuration
    int lane_id = warp::laneid();
    int warp_id = warpgroup::warpid();
    int warpgroup_id = warpgroup::groupid();

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    static_assert(sizeof(globals::A_fp8_tile) + 1024 +
                  sizeof(globals::B_fp8_tile) + 1024 +
                  sizeof(globals::C_tile) + 1024 <= config::DYNAMIC_SHARED_MEMORY);
    globals::A_fp8_tile &A_fp8_tile = allocator.allocate<globals::A_fp8_tile>();
    globals::B_fp8_tile &B_fp8_tile = allocator.allocate<globals::B_fp8_tile>();
    globals::C_tile &C_tile = allocator.allocate<globals::C_tile>();

    // Allocate tensor memory
    __shared__ uint32_t tm_addr_shared;
    uint32_t tm_addr = 0;
    if (warpid() == 0) {
        asm volatile("{tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;}"
            :: "l"(reinterpret_cast<uint64_t>(&tm_addr_shared)), "n"(512)); // assign max
        asm volatile("{tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;}");
    }
    __syncthreads();
    tm_addr = tm_addr_shared;

    // Set up mbarriers
    __shared__ semaphore inputs_arrived;
    __shared__ semaphore outputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1); // smem input arrival
        init_semaphore(outputs_arrived, 0, 1);
    }
    __syncthreads();

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 3 && lane_id == 0) {
            // Load input matrices to shared memory
            tma::expect_bytes(inputs_arrived, sizeof(globals::A_fp8_tile) + sizeof(globals::B_fp8_tile));
            tma::load_async(A_fp8_tile, G.A_fp8, {0, 0}, inputs_arrived);
            tma::load_async(B_fp8_tile, G.B_fp8, {0, 0}, inputs_arrived);
        } else if (warp_id == 0 && lane_id == 0) {
            // Launch tensor core matrix multiply
            constexpr uint64_t M = globals::ROW_BLOCK;
            constexpr uint64_t N = globals::COL_BLOCK;
            constexpr uint64_t K = globals::REDUCTION_BLOCK;
            constexpr uint32_t i_desc =
                (0b000 << 0)     | // dense matrix multiply
                (0b0 << 3)       | // no integer saturation needed
                (0b01 << 4)      | // FP32 accumulation
                (0b0 << 6)       | // SBZ
                (0b000 << 7)     | // Matrix A is E4M3
                (0b000 << 10)    | // Matrix B is E4M3
                (0b0 << 13)      | // Do not negate A
                (0b0 << 14)      | // Do not negate B
                (0b0 << 15)      | // Do not transpose A
                (0b0 << 16)      | // Do not transpose B
                ((N >> 3) << 17) | // N, encoded
                (0b0 << 23)      | // SBZ
                ((M >> 4) << 24) | // M, encoded
                (0b0 << 29)      | // SBZ
                (0b00 << 30);      // No shift in B
            kittens::st_descriptor<globals::A_fp8_tile, 0> a_desc(A_fp8_tile);
            kittens::st_descriptor<globals::B_fp8_tile, 0> b_desc(B_fp8_tile);
            wait(inputs_arrived, 0);
            asm volatile("{tcgen05.fence::after_thread_sync;}");
            asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory");
            asm volatile("{.reg .pred P1;                                              \t\n"
                         "setp.eq.u32 P1, 1, %4;                                       \t\n"
                         "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, P1;  \t\n}"
                         :: "r"(tm_addr), 
                            "l"(a_desc.chunk_descriptor(0)),
                            "l"(b_desc.chunk_descriptor(0)),
                            "r"(i_desc),
                            "n"(0));
            #pragma unroll
            for (int i = 1; i < K / 32; i++) {
                asm volatile("{.reg .pred P1;                                              \t\n"
                             "setp.eq.u32 P1, 1, %4;                                       \t\n"
                             "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, P1;  \t\n}"
                             :: "r"(tm_addr), 
                                "l"(a_desc.chunk_descriptor(i)),
                                "l"(b_desc.chunk_descriptor(i)),
                                "r"(i_desc),
                                "n"(1));
            }
            kittens::detail::tcgen05::commit<1>(outputs_arrived);
        }
    } else if (warpgroup_id == 0) {
        // Consumer group
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        // Wait for the matmul to complete
        wait(outputs_arrived, 0);

        // Load the output from tensor memory into registers
        tt<float, 128, 128> tm(tm_addr);
        rt_fl<globals::ROW_BLOCK / 4, globals::COL_BLOCK> C_reg;
        warpgroup::load_async(C_reg, tm);
        tensor_load_wait();

        // Store back to global memory
        warpgroup::store(C_tile, C_reg);
        warpgroup::sync(1);
        warpgroup::tma::store_async(G.C, C_tile, {0, 0});
        warpgroup::tma::store_async_read_wait();
        warpgroup::sync(1);
    }

    __syncthreads();
    if (warpid() == 0) {
        asm volatile("{tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;}"
            :: "r"(tm_addr), "n"(512));
    }
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel>(m, "kernel",
        &globals::A_fp8,
        &globals::B_fp8,
        &globals::C
    );
}
