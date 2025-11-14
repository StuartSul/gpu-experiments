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
    static constexpr int SCALE_BLOCK = 512; // 4 K=32-blocks for 128 rows

    using A_fp8_tile = st_fp8e4m3<ROW_BLOCK, REDUCTION_BLOCK>;
    using A_sc_vec = sv<fp8e8m0, SCALE_BLOCK>;
    using B_fp8_tile = st_fp8e4m3<COL_BLOCK, REDUCTION_BLOCK>;
    using B_sc_vec = sv<fp8e8m0, SCALE_BLOCK>;
    using C_tile = st_fl<ROW_BLOCK, COL_BLOCK>;

    gl<fp8e4m3, 1, 1, -1, -1, A_fp8_tile> A_fp8;        // M x K
    gl<fp8e8m0, 1, -1, -1, SCALE_BLOCK, A_sc_vec> A_sc; // (M // ROW_BLOCK) x (K // COL_BLOCK) x SCALE_BLOCK
    gl<fp8e4m3, 1, 1, -1, -1, B_fp8_tile> B_fp8;        // N x K
    gl<fp8e8m0, 1, 1, -1, SCALE_BLOCK, B_sc_vec> B_sc;  // (M // ROW_BLOCK) x (K // COL_BLOCK) x SCALE_BLOCK
    gl<float, 1, 1, -1, -1, C_tile> C;                  // M x N

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
                  sizeof(globals::A_sc_vec) + 1024 +
                  sizeof(globals::B_fp8_tile) + 1024 +
                  sizeof(globals::B_sc_vec) + 1024 +
                  sizeof(globals::C_tile) + 1024 <= config::DYNAMIC_SHARED_MEMORY);
    globals::A_fp8_tile &A_fp8_tile = allocator.allocate<globals::A_fp8_tile>();
    globals::A_sc_vec &A_sc_vec = allocator.allocate<globals::A_sc_vec>();
    globals::B_fp8_tile &B_fp8_tile = allocator.allocate<globals::B_fp8_tile>();
    globals::B_sc_vec &B_sc_vec = allocator.allocate<globals::B_sc_vec>();
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
    uint32_t out_tm_addr = tm_addr;
    uint32_t A_sc_tm_addr = tm_addr + 128;
    uint32_t B_sc_tm_addr = tm_addr + 128 + 64;

    // Set up mbarriers
    __shared__ semaphore inputs_arrived;
    __shared__ semaphore scale_arrived;
    __shared__ semaphore outputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 2); // smem input + tm scale arrival
        init_semaphore(scale_arrived, 0, 1);
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
        } else if (warp_id == 2 && lane_id == 0) {
            // Load scale matrices to shared memory
            tma::expect_bytes(scale_arrived, sizeof(globals::A_sc_vec) + sizeof(globals::B_sc_vec));
            tma::load_async(A_sc_vec, G.A_sc, {0, 0, 0}, scale_arrived);
            tma::load_async(B_sc_vec, G.B_sc, {0, 0, 0}, scale_arrived);
        } else if (warp_id == 1 && lane_id == 0) {
            // Load scale matrices to tensor memory
            wait(scale_arrived, 0);
            uint64_t A_sc_addr = reinterpret_cast<uint64_t>(__cvta_generic_to_shared(&A_sc_vec));
            uint64_t A_sc_desc = 
                (((A_sc_addr & 0x3FFFFULL) >> 4) << 0)  | // bits 00-13: smem addr
                (0ULL << 14)                         | // bits 14-15: SBZ
                (((128ULL & 0x3FFFFULL) >> 4) << 16) | // bits 16-29: leading dimension relative byte offset (doesn't seem to matter)
                (0ULL << 30)                         | // bits 30-31: SBZ
                (((128ULL & 0x3FFFFULL) >> 4) << 32) | // bits 32-45: stride dimension byte offset (16B atom per row x 8)
                (1ULL << 46)                         | // bits 46-48: fixed constant of 1
                (0ULL << 49)                         | // bits 49-51: matrix byte offset (0 if 1024B-aligned)
                (0ULL << 52)                         | // bits 52-52: leading dimension stride mode (0 for relative)
                (0ULL << 53)                         | // bits 53-60: fixed constant of 0
                (0ULL << 61);                          // bits 61-63: swizzling mode used (0 for no swizzling)
            uint64_t B_sc_addr = reinterpret_cast<uint64_t>(__cvta_generic_to_shared(&B_sc_vec));
            uint64_t B_sc_desc = 
                (((B_sc_addr & 0x3FFFFULL) >> 4) << 0)  | // bits 00-13: smem addr
                (0ULL << 14)                         | // bits 14-15: SBZ
                (((128ULL & 0x3FFFFULL) >> 4) << 16) | // bits 16-29: leading dimension relative byte offset (doesn't seem to matter)
                (0ULL << 30)                         | // bits 30-31: SBZ
                (((128ULL & 0x3FFFFULL) >> 4) << 32) | // bits 32-45: stride dimension byte offset (16B atom per row x 8)
                (1ULL << 46)                         | // bits 46-48: fixed constant of 1
                (0ULL << 49)                         | // bits 49-51: matrix byte offset (0 if 1024B-aligned)
                (0ULL << 52)                         | // bits 52-52: leading dimension stride mode (0 for relative)
                (0ULL << 53)                         | // bits 53-60: fixed constant of 0
                (0ULL << 61);                          // bits 61-63: swizzling mode used (0 for no swizzling)        
            asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}"
                :: "r"(A_sc_tm_addr), "l"(A_sc_desc));
            asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}"
                :: "r"(B_sc_tm_addr), "l"(B_sc_desc));
            kittens::detail::tcgen05::commit<1>(inputs_arrived);
            asm volatile("{tcgen05.fence::before_thread_sync;}");
        } else if (warp_id == 0 && lane_id == 0) {
            // Launch tensor core matrix multiply
            constexpr uint64_t M = globals::ROW_BLOCK;
            constexpr uint64_t N = globals::COL_BLOCK;
            constexpr uint64_t K = globals::REDUCTION_BLOCK;
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
            // Table 44: Instruction descriptor format for .kind::mxf8f6f4
            constexpr uint32_t I_desc = 
                (0b00 << 0)      | // SBZ
                (0b0 << 2)       | // Dense matrix multiply
                (0b0 << 3)       | // SBZ
                (0b00 << 4)      | // B_sc data ID (should be 0, 1, 2, 3)
                (0b0 << 6)       | // SBZ
                (0b000 << 7)     | // Matrix A is E4M3
                (0b000 << 10)    | // Matrix B is E4M3
                (0b0 << 13)      | // Do not negate A
                (0b0 << 14)      | // Do not negate B
                (0b0 << 15)      | // Do not transpose A
                (0b0 << 16)      | // Do not transpose B
                ((N >> 3) << 17) | // N, encoded
                (0b1 << 23)      | // Scale type is UE8M0
                (0b000 << 24)    | // SBZ                
                ((M >> 7) << 27) | // M, encoded
                (0b00 << 29)     | // A_sc data ID (should be 0, 1, 2, 3)
                (0b00 << 31);      // SBZ
            constexpr uint32_t I_descs[4] = {
                (I_desc & ~(0b11 << 4) & ~(0b11 << 29)) | (0b00 << 4) | (0b00 << 29),
                (I_desc & ~(0b11 << 4) & ~(0b11 << 29)) | (0b01 << 4) | (0b01 << 29),
                (I_desc & ~(0b11 << 4) & ~(0b11 << 29)) | (0b10 << 4) | (0b10 << 29),
                (I_desc & ~(0b11 << 4) & ~(0b11 << 29)) | (0b11 << 4) | (0b11 << 29)
            };
            kittens::st_descriptor<globals::A_fp8_tile, 0> A_desc(A_fp8_tile);
            kittens::st_descriptor<globals::B_fp8_tile, 0> B_desc(B_fp8_tile);
            wait(inputs_arrived, 0);
            asm volatile("{tcgen05.fence::after_thread_sync;}");
            asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory");
            asm volatile("{.reg .pred P1; \t\n"
                         "setp.eq.u32 P1, 1, %6; \t\n"
                         "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %5, [%3], [%4], P1; \t\n}"
                         :: "r"(out_tm_addr), 
                            "l"(A_desc.chunk_descriptor(0)),
                            "l"(B_desc.chunk_descriptor(0)),
                            "r"(A_sc_tm_addr),
                            "r"(B_sc_tm_addr),
                            "r"(I_descs[0]),
                            "n"(0));
            #pragma unroll
            for (int i = 1; i < K / 32; i++) {
                asm volatile("{.reg .pred P1; \t\n"
                             "setp.eq.u32 P1, 1, %6; \t\n"
                             "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %5, [%3], [%4], P1; \t\n}"
                             :: "r"(out_tm_addr), 
                                "l"(A_desc.chunk_descriptor(i)),
                                "l"(B_desc.chunk_descriptor(i)),
                                "r"(A_sc_tm_addr),
                                "r"(B_sc_tm_addr),
                                "r"(I_descs[i]),
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
        tt<float, 128, 128> tm(out_tm_addr);
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
        &globals::A_sc,
        &globals::B_fp8,
        &globals::B_sc,
        &globals::C
    );
}
