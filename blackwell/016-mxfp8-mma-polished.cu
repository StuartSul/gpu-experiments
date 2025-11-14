/*
    Just a refactor to easily expand to other functionalities. But found some interesting things

    Observations:
        - Increasing the PIPELINE_STAGES from 3 to 4 gives around 300 more TFLOPs (1510 -> 1840)
        - Just cleaning up and encapsulating things inside a function gives 100 more TFLOPs!!
            - Probably because it helps the compiler optimize away certain data flow, as well
              as less code -> better i-cache reuse
        - Now at 1945 TFLOPs

    More observations:
        - Separating the first matmul to set accumulate flag at compile time has no benefit. Just
          makes the code less simple.
        - Actually, the kernel is about 5 TFLOPs faster without such optimization(!)
            - Again, less code = easier for compiler to optimize & better i-cache reuse
        - Cleaner code and now at ** 1950 TFLOPs **

    Even more observations:
        - Using bfloat16 for C is about about 20 more TFLOPs than float32
        - Thus minor, it only makes sense to do it for memory savings
*/

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
    static constexpr int SM_COUNT = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 2;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;

    static constexpr int PIPELINE_STAGES = 4;
};

// Kernel globals
struct globals {
    static constexpr int SUPERGROUP_BLOCKS = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 128;
    static constexpr int REDUCTION_BLOCK = 128;
    static constexpr int SCALE_BLOCK = 512; // 4 K=32-blocks for 128 rows/cols

    using A_fp8_tile = st_fp8e4m3<ROW_BLOCK, REDUCTION_BLOCK>;
    using A_sc_vec = sv<fp8e8m0, SCALE_BLOCK>;
    using B_fp8_tile = st_fp8e4m3<COL_BLOCK, REDUCTION_BLOCK>;
    using B_sc_vec = sv<fp8e8m0, SCALE_BLOCK>;
    using C_tile = st_fl<ROW_BLOCK / 8, COL_BLOCK>;

    gl<fp8e4m3, 1, 1, -1, -1, A_fp8_tile> A;            // M x K
    gl<fp8e8m0, 1, -1, -1, SCALE_BLOCK, A_sc_vec> A_sc; // (M // ROW_BLOCK) x (K // COL_BLOCK) x SCALE_BLOCK
    gl<fp8e4m3, 1, 1, -1, -1, B_fp8_tile> B;            // N x K
    gl<fp8e8m0, 1, -1, -1, SCALE_BLOCK, B_sc_vec> B_sc;  // (M // ROW_BLOCK) x (K // COL_BLOCK) x SCALE_BLOCK
    gl<float, 1, 1, -1, -1, C_tile> C;                  // M x N

    __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }

    struct pipeline_input_tiles {
        A_fp8_tile A;
        B_fp8_tile B;
    };

    struct pipeline_input_scales {
        A_sc_vec A;
        B_sc_vec B;
    };

    struct pipeline_outputs {
        C_tile C;
    };
};

#define PIPELINE_INIT \
    int num_blocks_per_row = G.C.cols() / globals::COL_BLOCK; \
    int num_blocks_per_col = G.C.rows() / globals::ROW_BLOCK; \
    int num_blocks = num_blocks_per_row * num_blocks_per_col; \
    int num_iters_per_block = G.A.cols() / globals::REDUCTION_BLOCK; \
    int num_blocks_per_supergroup = globals::SUPERGROUP_BLOCKS * num_blocks_per_row;

#define WORKER_OUTER_LOOP(...) \
do { \
    for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) { \
        int supergroup_idx = block_idx / num_blocks_per_supergroup; \
        int idx_within_supergroup = block_idx % num_blocks_per_supergroup; \
        int rows_in_supergroup = min(globals::SUPERGROUP_BLOCKS, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_BLOCKS); \
        int row_within_supergroup = idx_within_supergroup % rows_in_supergroup; \
        [[maybe_unused]] int row_block_idx = supergroup_idx * globals::SUPERGROUP_BLOCKS + row_within_supergroup; \
        [[maybe_unused]] int col_block_idx = idx_within_supergroup / rows_in_supergroup; \
        __VA_ARGS__ \
    } \
} while (0)

#define WORKER_INNER_LOOP(...) \
do { \
    for (int i = 0; i < num_iters_per_block; i++) { \
        __VA_ARGS__ \
        stage = (stage + 1) % config::PIPELINE_STAGES; \
    } \
} while (0)

__device__ static inline uint32_t allocate_tensor_memory() {
    __shared__ uint32_t tm_addr_shared;
    if (warpid() == 0) {
        asm volatile("{tcgen05.alloc.cta_group::1.sync.aligned.b32 [%0], %1;}"
            :: "l"(reinterpret_cast<uint64_t>(&tm_addr_shared)), "n"(512)); // assign max TM
        asm volatile("{tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;}");
    }
    __syncthreads();
    uint32_t tm_addr = tm_addr_shared;
    return tm_addr;
}

__device__ static inline void deallocate_tensor_memory(uint32_t tm_addr) {
    __syncthreads();
    if (warpid() == 0) {
        asm volatile("{tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;}"
            :: "r"(tm_addr), "n"(512));
    }
}

__device__ static inline void load_scales_to_tensor_memory(uint32_t tm_addr, fp8e8m0 *sc_addr) {
    uint64_t sc_smem_addr = reinterpret_cast<uint64_t>(__cvta_generic_to_shared(sc_addr));
    uint64_t sc_desc = 
        (((sc_smem_addr & 0x3FFFFULL) >> 4) << 0) | // bits 00-13: smem addr
        (0ULL << 14)                              | // bits 14-15: SBZ
        (((128ULL & 0x3FFFFULL) >> 4) << 16)      | // bits 16-29: leading dimension relative byte offset (doesn't seem to matter)
        (0ULL << 30)                              | // bits 30-31: SBZ
        (((128ULL & 0x3FFFFULL) >> 4) << 32)      | // bits 32-45: stride dimension byte offset (16B atom per row x 8)
        (1ULL << 46)                              | // bits 46-48: fixed constant of 1
        (0ULL << 49)                              | // bits 49-51: matrix byte offset (0 if 1024B-aligned)
        (0ULL << 52)                              | // bits 52-52: leading dimension stride mode (0 for relative)
        (0ULL << 53)                              | // bits 53-60: fixed constant of 0
        (0ULL << 61);                               // bits 61-63: swizzling mode used (0 for no swizzling)  
    asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}" :: "r"(tm_addr), "l"(sc_desc));
}

__device__ static inline void commit_async_tcgen05_op(semaphore &sem) {
    kittens::detail::tcgen05::commit<1>(sem);
    asm volatile("{tcgen05.fence::before_thread_sync;}");
}

template <ducks::st::all A_ST, ducks::st::all B_ST>
__device__ static inline void launch_128x128_mxfp8_matmul(
    uint32_t out_tm_addr, A_ST &A, B_ST &B, uint32_t A_sc_tm_addr, uint32_t B_sc_tm_addr, int accumulate
) {
    constexpr uint64_t M = 128;
    constexpr uint64_t N = 128;
    constexpr uint64_t K = 128;

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

    kittens::st_descriptor<A_ST, 0> A_desc(A);
    kittens::st_descriptor<B_ST, 0> B_desc(B);

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
                    "r"(accumulate));
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
}

// Kernel implementation
__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    static_assert(sizeof(globals::pipeline_input_tiles) * config::PIPELINE_STAGES +
                  sizeof(globals::pipeline_input_scales) * config::PIPELINE_STAGES + 1024 +
                  sizeof(globals::C_tile) <= config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_input_tiles (&input_tiles)[config::PIPELINE_STAGES] = allocator.allocate<globals::pipeline_input_tiles, config::PIPELINE_STAGES>();
    globals::pipeline_input_scales (&input_scales)[config::PIPELINE_STAGES] = allocator.allocate<globals::pipeline_input_scales, config::PIPELINE_STAGES>();
    globals::pipeline_outputs &output_tiles = allocator.allocate<globals::pipeline_outputs>();

    // Allocate tensor memory
    uint32_t tm_addr = allocate_tensor_memory();
    uint32_t out_tm_addr[2] = {tm_addr, tm_addr + 128};
    uint32_t A_sc_tm_addr[config::PIPELINE_STAGES] = {
        tm_addr + 256, tm_addr + 256 + 32, tm_addr + 256 + 64, tm_addr + 256 + 96};
    uint32_t B_sc_tm_addr[config::PIPELINE_STAGES] = {
        tm_addr + 384, tm_addr + 384 + 32, tm_addr + 384 + 64, tm_addr + 384 + 96};

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[config::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[config::PIPELINE_STAGES];
    __shared__ semaphore scales_arrived[config::PIPELINE_STAGES];
    __shared__ semaphore scales_finished[config::PIPELINE_STAGES];
    __shared__ semaphore tensors_finished;
    __shared__ semaphore outputs_arrived;
    if (threadIdx.x == 0) {
        for (int i = 0; i < config::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 2); // smem input + tm scale arrival
            init_semaphore(inputs_finished[i], 0, 1);
            init_semaphore(scales_arrived[i], 0, 1);
            init_semaphore(scales_finished[i], 0, 1);
        }
        init_semaphore(tensors_finished, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
    }
    __syncthreads();

    // Warpgroup configuration
    int lane_id = warp::laneid();
    int warp_id = warpgroup::warpid();
    int warpgroup_id = warpgroup::groupid();

    // Pipeline configuration
    PIPELINE_INIT

    // Declare stage and phasebits for semaphore waits
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    int last_stage = -1;

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 3 && lane_id == 0) {
            // Input tile loader
            WORKER_OUTER_LOOP({
                WORKER_INNER_LOOP({
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    if (stage == last_stage) {
                        arrive(outputs_arrived);
                        last_stage = -1;
                    }
                    tma::expect_bytes(inputs_arrived[stage], sizeof(globals::A_fp8_tile) + sizeof(globals::B_fp8_tile));
                    tma::load_async(input_tiles[stage].A, G.A, {row_block_idx, i}, inputs_arrived[stage]);
                    tma::load_async(input_tiles[stage].B, G.B, {col_block_idx, i}, inputs_arrived[stage]);
                    if (i == num_iters_per_block - 1) {
                        last_stage = stage;
                    }
                });
            });
            wait(inputs_finished[last_stage], get_phasebit<1>(phasebits, last_stage));
            arrive(outputs_arrived);
        } else if (warp_id == 2 && lane_id == 0) {
            // Input scale loader
            WORKER_OUTER_LOOP({
                WORKER_INNER_LOOP({
                    wait(scales_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::expect_bytes(scales_arrived[stage], sizeof(globals::A_sc_vec) + sizeof(globals::B_sc_vec));
                    tma::load_async(input_scales[stage].A, G.A_sc, {row_block_idx, i, 0}, scales_arrived[stage]);
                    tma::load_async(input_scales[stage].B, G.B_sc, {col_block_idx, i, 0}, scales_arrived[stage]);
                });
            });
        } else if (warp_id == 1 && lane_id == 0) {
            // Input scale tensor memory loader
            WORKER_OUTER_LOOP({
                WORKER_INNER_LOOP({
                    wait(scales_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    load_scales_to_tensor_memory(A_sc_tm_addr[stage], &input_scales[stage].A[0]);
                    load_scales_to_tensor_memory(B_sc_tm_addr[stage], &input_scales[stage].B[0]);
                    commit_async_tcgen05_op(inputs_arrived[stage]);
                });
            });
        } else if (warp_id == 0 && lane_id == 0) {
            // Tensor core matrix multiply launcher
            WORKER_OUTER_LOOP({
                WORKER_INNER_LOOP({
                    if (i == 0) {
                        wait(tensors_finished, get_phasebit<1>(phasebits, config::PIPELINE_STAGES));
                        update_phasebit<1>(phasebits, config::PIPELINE_STAGES);
                    }
                    wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    arrive(scales_finished[stage]);
                    launch_128x128_mxfp8_matmul(out_tm_addr[0], 
                                                input_tiles[stage].A, input_tiles[stage].B, 
                                                A_sc_tm_addr[stage], B_sc_tm_addr[stage],
                                                i == 0 ? 0 : 1);
                    commit_async_tcgen05_op(inputs_finished[stage]);
                });
            });
        }
    } else {
        // Consumer group
        using consumer = group<config::CONSUMER_WARPGROUPS * WARPGROUP_WARPS>;
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        WORKER_OUTER_LOOP({
            wait(outputs_arrived, get_phasebit<0>(phasebits, config::PIPELINE_STAGES));
            update_phasebit<0>(phasebits, config::PIPELINE_STAGES);

            // Load the output from tensor memory into registers
            tt<float, 128, 128> tm(out_tm_addr[0]);
            rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_reg;
            consumer::load_async(C_reg, tm);
            tensor_load_wait();
            consumer::sync(1);
            consumer::arrive(tensors_finished);

            // Store back to global memory
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (warp_id * 2 + warpgroup_id == i) {
                    warp::store(output_tiles.C, C_reg);
                    __syncwarp();
                    if (lane_id == 0) {
                        tma::store_async(G.C, output_tiles.C, {row_block_idx * 8 + i, col_block_idx});
                        tma::store_async_read_wait();
                    }
                }
                consumer::sync(1);
            }
        });
    }

    deallocate_tensor_memory(tm_addr);
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel>(m, "kernel",
        &globals::A,
        &globals::A_sc,
        &globals::B,
        &globals::B_sc,
        &globals::C
    );
}
