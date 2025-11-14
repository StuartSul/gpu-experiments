/*
    *Verified that this emits the same TFLOPs as 46
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct mxfp8_matmul_pipeline {

    // Pipeline stages
    static constexpr int PIPELINE_STAGES = 5;

    // Per-block dimensions
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 128; // reduction axis
    static constexpr int SCALE_BLOCK = 512; // 4 K=32 blocks per 128 rows per CTA

    // Supergrouping for higher L2 utilization
    static constexpr int SUPERGROUP_SIZE = 12;

    // Producer warp specialization
    static constexpr int TILE_LOADER = 3;
    static constexpr int SCALE_SM_LOADER = 2;
    static constexpr int SCALE_TM_LOADER = 1;
    static constexpr int MMA_LAUNCHER = 0;

    // Tensor memory addresses
    static constexpr uint32_t tm_output_offset = 0;
    static constexpr uint32_t tm_A_sc_offset = 256;
    static constexpr uint32_t tm_B_sc_offset = 256 + 128;

    // Kernel configuration
    struct config {
        static constexpr int CLUSTER_SIZE = 2;

        static constexpr int SM_COUNT = 148;
        static constexpr int STATIC_SHARED_MEMORY = 1024;
        static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

        static constexpr int NUM_CONSUMERS = 2;
        static constexpr int NUM_PRODUCERS = 1;
        static constexpr int NUM_WARPGROUPS = NUM_CONSUMERS + NUM_PRODUCERS;
        static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
        static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

        static constexpr int PRODUCER_REGISTERS = 56;
        static constexpr int CONSUMER_REGISTERS = 224;
    };

    // Type aliases
    using A_tile = st_fp8e4m3<ROW_BLOCK / 2, RED_BLOCK>; // CTA distributed
    using B_tile = st_fp8e4m3<COL_BLOCK / 2, RED_BLOCK>; // CTA distributed
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK / 2>;  // CTA/WG distributed
    using A_sc_vec = sv<fp8e8m0, SCALE_BLOCK>;
    using B_sc_vec = sv<fp8e8m0, SCALE_BLOCK>;
    using consumer = group<config::NUM_CONSUMERS * WARPGROUP_WARPS>;
    using tm_t = tt<float, ROW_BLOCK / 2, COL_BLOCK>;

    struct pipeline_input_tiles {
        A_tile A;
        B_tile B;
    };

    struct pipeline_input_scales {
        A_sc_vec A;
        B_sc_vec B[2];
    };

    struct pipeline_outputs {
        C_tile C;
    };

    // Pipeline state
    struct state {
        const int cta_id;
        const int warpgroup_id;
        const int warp_id;
        const int lane_id;

        int row_block_idx;
        int col_block_idx;
        int red_block_start;
        int red_block_end;

        pipeline_input_tiles (&input_tiles)[PIPELINE_STAGES];
        pipeline_input_scales (&input_scales)[PIPELINE_STAGES];
        pipeline_outputs &outputs;

        const uint32_t tm_addr;

        semaphore (&tiles_arrived)[PIPELINE_STAGES];
        semaphore (&scales_sm_arrived)[PIPELINE_STAGES];
        semaphore (&scales_tm_arrived)[PIPELINE_STAGES];
        semaphore (&matmul_finished)[PIPELINE_STAGES];
        semaphore &tensor_finished;
        semaphore &outputs_arrived;

        uint32_t stage;
        uint32_t pending_stages;
        uint32_t phasebits;
    };

    template <bool load_A, bool load_B, typename globals>
    __device__ static inline void tile_loader_loop(const globals &G, state &S) {
        constexpr int bytes_per_iter = (load_A ? sizeof(A_tile) : 0) + (load_B ? sizeof(B_tile) : 0);

        for (int red_block_idx = S.red_block_start; red_block_idx < S.red_block_end; red_block_idx++) {
            tma::cluster::wait(S.matmul_finished[S.stage], get_phasebit<1>(S.phasebits, S.stage));
            update_phasebit<1>(S.phasebits, S.stage);

            if ((S.pending_stages >> S.stage) & 0b1) {
                arrive(S.outputs_arrived);
                S.pending_stages &= ~(0b1 << S.stage); // clear the bit
            }

            // Load to current CTA, but only signal mbarrier at CTA 0 (tma::cluster is purely for cluster-level synchronization)
            if constexpr (load_A || load_B) {
                tma::cluster::expect_bytes(S.tiles_arrived[S.stage], bytes_per_iter, 0);
                if constexpr (load_A)
                    tma::cluster::load_async(S.input_tiles[S.stage].A, G.A, {S.row_block_idx * 2 + S.cta_id, red_block_idx}, S.tiles_arrived[S.stage], (uint16_t)(1 << S.cta_id), 0); 
                if constexpr (load_B)
                    tma::cluster::load_async(S.input_tiles[S.stage].B, G.B, {S.col_block_idx * 2 + S.cta_id, red_block_idx}, S.tiles_arrived[S.stage], (uint16_t)(1 << S.cta_id), 0);
            } else {
                tma::cluster::arrive(S.tiles_arrived[S.stage], 0, 1); // signal immediately
            }

            if (red_block_idx == S.red_block_end - 1) {
                S.pending_stages |= 0b1 << S.stage; // set the bit
            }

            // Update stage
            S.stage = (S.stage + 1) % PIPELINE_STAGES;
        }
    }

    template <bool do_load, bool load_B1, typename globals>
    __device__ static inline void scale_sm_loader_loop(const globals &G, state &S) {
        constexpr int bytes_per_iter = do_load ? (load_B1 ? sizeof(pipeline_input_scales) : sizeof(A_sc_vec) + sizeof(B_sc_vec)) : 0;

        // Load scale matrices to shared memory
        for (int red_block_idx = S.red_block_start; red_block_idx < S.red_block_end; red_block_idx++) {
            tma::cluster::wait(S.scales_tm_arrived[S.stage], get_phasebit<1>(S.phasebits, S.stage));
            update_phasebit<1>(S.phasebits, S.stage);

            if constexpr (do_load) {
                tma::expect_bytes(S.scales_sm_arrived[S.stage], bytes_per_iter);
                tma::load_async(S.input_scales[S.stage].A, G.A_sc, {S.row_block_idx * 2 + S.cta_id, red_block_idx, 0}, S.scales_sm_arrived[S.stage]);
                tma::load_async(S.input_scales[S.stage].B[0], G.B_sc, {S.col_block_idx * 2 + 0, red_block_idx, 0}, S.scales_sm_arrived[S.stage]);
                if constexpr (load_B1)
                    tma::load_async(S.input_scales[S.stage].B[1], G.B_sc, {S.col_block_idx * 2 + 1, red_block_idx, 0}, S.scales_sm_arrived[S.stage]);
            } else {
                arrive(S.scales_sm_arrived[S.stage]);
            }

            // Update stage
            S.stage = (S.stage + 1) % PIPELINE_STAGES;
        }
    }

    template <typename globals>
    __device__ static inline void scale_tm_loader_loop(const globals &G, state &S) {
        // Load scale matrices to tensor memory
        for (int red_block_idx = S.red_block_start; red_block_idx < S.red_block_end; red_block_idx++) {
            wait(S.scales_sm_arrived[S.stage], get_phasebit<0>(S.phasebits, S.stage));
            update_phasebit<0>(S.phasebits, S.stage);
            tma::cluster::wait(S.matmul_finished[S.stage], get_phasebit<1>(S.phasebits, S.stage));
            update_phasebit<1>(S.phasebits, S.stage);

            // Prepare shared memory descriptors
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
            uint64_t A_sc_addr = reinterpret_cast<uint64_t>(__cvta_generic_to_shared(&S.input_scales[S.stage].A));
            uint64_t A_sc_desc = 
                (((A_sc_addr & 0x3FFFFULL) >> 4) << 0) | // bits 00-13: smem addr
                (0ULL << 14)                           | // bits 14-15: SBZ
                (((128ULL & 0x3FFFFULL) >> 4) << 16)   | // bits 16-29: leading dimension relative byte offset (doesn't seem to matter)
                (0ULL << 30)                           | // bits 30-31: SBZ
                (((128ULL & 0x3FFFFULL) >> 4) << 32)   | // bits 32-45: stride dimension byte offset (16B atom per row x 8)
                (1ULL << 46)                           | // bits 46-48: fixed constant of 1
                (0ULL << 49)                           | // bits 49-51: matrix byte offset (0 if 1024B-aligned)
                (0ULL << 52)                           | // bits 52-52: leading dimension stride mode (0 for relative)
                (0ULL << 53)                           | // bits 53-60: fixed constant of 0
                (0ULL << 61);                            // bits 61-63: swizzling mode used (0 for no swizzling)
            uint64_t B_sc_addr[2] = {reinterpret_cast<uint64_t>(__cvta_generic_to_shared(&S.input_scales[S.stage].B[0])),
                                     reinterpret_cast<uint64_t>(__cvta_generic_to_shared(&S.input_scales[S.stage].B[1]))};
            constexpr uint64_t B_sc_desc_base = 
                (0ULL << 0)                            | // bits 00-13: smem addr (filled in below)
                (0ULL << 14)                           | // bits 14-15: SBZ
                (((128ULL & 0x3FFFFULL) >> 4) << 16)   | // bits 16-29: leading dimension relative byte offset (doesn't seem to matter)
                (0ULL << 30)                           | // bits 30-31: SBZ
                (((128ULL & 0x3FFFFULL) >> 4) << 32)   | // bits 32-45: stride dimension byte offset (16B atom per row x 8)
                (1ULL << 46)                           | // bits 46-48: fixed constant of 1
                (0ULL << 49)                           | // bits 49-51: matrix byte offset (0 if 1024B-aligned)
                (0ULL << 52)                           | // bits 52-52: leading dimension stride mode (0 for relative)
                (0ULL << 53)                           | // bits 53-60: fixed constant of 0
                (0ULL << 61);                            // bits 61-63: swizzling mode used (0 for no swizzling)   
            uint64_t B_sc_desc[2] = {B_sc_desc_base | (((B_sc_addr[0] & 0x3FFFFULL) >> 4) << 0),
                                     B_sc_desc_base | (((B_sc_addr[1] & 0x3FFFFULL) >> 4) << 0)};

            // Perform the tensor memory loads
            asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}"
                :: "r"(S.tm_addr + tm_A_sc_offset + 16 * S.stage), "l"(A_sc_desc));
            asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}"
                :: "r"(S.tm_addr + tm_B_sc_offset + 16 * S.stage + 4 * 0), "l"(B_sc_desc[0]));
            asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}"
                :: "r"(S.tm_addr + tm_B_sc_offset + 16 * S.stage + 4 * 1), "l"(B_sc_desc[1]));
            asm volatile("{tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;}"
                :: "l"(__cvta_generic_to_shared(&S.scales_tm_arrived[S.stage])), "h"((uint16_t)(0b11))); // signal both CTAs
            asm volatile("{tcgen05.fence::before_thread_sync;}");

            // Update stage
            S.stage = (S.stage + 1) % PIPELINE_STAGES;
        }
    }

    template <typename globals>
    __device__ static inline void mma_launcher_loop(const globals &G, state &S) {
        // Launch tensor core matrix multiplies
        tma::cluster::wait(S.tensor_finished, get_phasebit<1>(S.phasebits, PIPELINE_STAGES));
        update_phasebit<1>(S.phasebits, PIPELINE_STAGES);

        for (int red_block_idx = S.red_block_start; red_block_idx < S.red_block_end; red_block_idx++) {
            tma::cluster::wait(S.tiles_arrived[S.stage], get_phasebit<0>(S.phasebits, S.stage));
            tma::cluster::wait(S.scales_tm_arrived[S.stage], get_phasebit<0>(S.phasebits, S.stage));
            update_phasebit<0>(S.phasebits, S.stage); // update phasebit after both waits

            // Convenience aliases
            constexpr uint64_t M = ROW_BLOCK;
            constexpr uint64_t N = COL_BLOCK;
            constexpr uint64_t K = RED_BLOCK;

            // Prepare the instruction descriptor
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

            // Launch the tensor core MXFP8 matrix multiplies
            kittens::st_descriptor<A_tile, 0> A_desc(S.input_tiles[S.stage].A);
            kittens::st_descriptor<B_tile, 0> B_desc(S.input_tiles[S.stage].B);
            asm volatile("{tcgen05.fence::after_thread_sync;}");
            asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory");
            asm volatile("{.reg .pred P1; \t\n"
                         "setp.eq.u32 P1, 1, %6; \t\n"
                         "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %5, [%3], [%4], P1; \t\n}"
                         :: "r"(S.tm_addr + tm_output_offset), 
                            "l"(A_desc.chunk_descriptor(0)),
                            "l"(B_desc.chunk_descriptor(0)),
                            "r"(S.tm_addr + tm_A_sc_offset + 16 * S.stage),
                            "r"(S.tm_addr + tm_B_sc_offset + 16 * S.stage),
                            "r"(I_descs[0]),
                            "r"(red_block_idx == S.red_block_start ? 0 : 1));
            #pragma unroll
            for (int i = 1; i < K / 32; i++) {
                asm volatile("{.reg .pred P1; \t\n"
                             "setp.eq.u32 P1, 1, %6; \t\n"
                             "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X [%0], %1, %2, %5, [%3], [%4], P1; \t\n}"
                             :: "r"(S.tm_addr + tm_output_offset), 
                                "l"(A_desc.chunk_descriptor(i)),
                                "l"(B_desc.chunk_descriptor(i)),
                                "r"(S.tm_addr + tm_A_sc_offset + 16 * S.stage),
                                "r"(S.tm_addr + tm_B_sc_offset + 16 * S.stage),
                                "r"(I_descs[i]),
                                "n"(1));
            }
            kittens::detail::tcgen05::commit<config::CLUSTER_SIZE>(S.matmul_finished[S.stage]);

            // Update stage
            S.stage = (S.stage + 1) % PIPELINE_STAGES;
        }
    }

    template <bool do_store, bool store_C1, typename globals>
    __device__ static inline void consumer_loop(const globals &G, state &S) {
        // Wait for the matmul to complete
        wait(S.outputs_arrived, get_phasebit<0>(S.phasebits, PIPELINE_STAGES));
        update_phasebit<0>(S.phasebits, PIPELINE_STAGES);

        // Load the output from tensor memory into registers
        tm_t tm(S.tm_addr + tm_output_offset);
        rt_bf<ROW_BLOCK / 8, COL_BLOCK / 2> C_reg;
        if (S.red_block_start >= S.red_block_end) {
            warp::zero(C_reg);
        } else {
            warpgroup::load_async(C_reg, tm.subtile<tt<float, ROW_BLOCK / 2, COL_BLOCK / 2>>(0, S.warpgroup_id * COL_BLOCK / 2));
            tensor_load_wait();
            consumer::sync(1);
            if (consumer::laneid() == 0)
                tma::cluster::arrive(S.tensor_finished, 0, 1); // signal CTA 0
        }

        constexpr int num_store_iters = do_store ? (store_C1 ? 2 : 1) : 0;

        #pragma unroll
        for (int i = 0; i < num_store_iters; i++) {
            if (S.warpgroup_id == i) {
                warpgroup::store(S.outputs.C, C_reg);
                warpgroup::sync(2 + i);
                if (warpgroup::laneid() == 0) {
                    tma::store_async(G.C, S.outputs.C, {S.row_block_idx * 2 + S.cta_id, S.col_block_idx * 2 + i});
                    tma::store_async_read_wait();
                }
            }
            consumer::sync(1);
        }
    }

    template <typename main_loop_policy, typename globals>
    __device__ static inline void dispatcher(const globals &G) {
        // Declare shared memory
        extern __shared__ int __shm[]; 
        tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    
        // Allocate shared memory
        static_assert(sizeof(pipeline_input_tiles) * PIPELINE_STAGES + 1024 +
                      sizeof(pipeline_input_scales) * PIPELINE_STAGES + 1024 +
                      sizeof(pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY);
        pipeline_input_tiles (&input_tiles)[PIPELINE_STAGES] = sm_allocator.allocate<pipeline_input_tiles, PIPELINE_STAGES>();
        pipeline_input_scales (&input_scales)[PIPELINE_STAGES] = sm_allocator.allocate<pipeline_input_scales, PIPELINE_STAGES>();
        pipeline_outputs &outputs = sm_allocator.allocate<pipeline_outputs>();
    
        // Allocate tensor memory
        __shared__ uint32_t tm_addr_shared;
        if (warpid() == 0) {
            asm volatile("{tcgen05.alloc.cta_group::2.sync.aligned.b32 [%0], %1;}"
                :: "l"(reinterpret_cast<uint64_t>(&tm_addr_shared)), "n"(512)); // assign max
            asm volatile("{tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;}");
        }

        // Initialize mbarriers
        __shared__ semaphore tiles_arrived[PIPELINE_STAGES];
        __shared__ semaphore scales_sm_arrived[PIPELINE_STAGES];
        __shared__ semaphore scales_tm_arrived[PIPELINE_STAGES];
        __shared__ semaphore matmul_finished[PIPELINE_STAGES];
        __shared__ semaphore tensor_finished;
        __shared__ semaphore outputs_arrived;
        if (threadIdx.x == 32) {
            #pragma unroll
            for (int i = 0; i < PIPELINE_STAGES; ++i) {
                init_semaphore(tiles_arrived[i], 0, config::CLUSTER_SIZE);
                init_semaphore(scales_sm_arrived[i], 0, 1); // local
                init_semaphore(scales_tm_arrived[i], 0, config::CLUSTER_SIZE);
                init_semaphore(matmul_finished[i], 0, 1); // odd CTA
            }
            init_semaphore(tensor_finished, 0, config::CLUSTER_SIZE);
            init_semaphore(outputs_arrived, 0, 1); // local
        }
        asm volatile("{barrier.cluster.arrive.release.aligned;}");
        asm volatile("{barrier.cluster.wait.acquire.aligned;}");

        // Set up matmul pipeline state
        state S {
            .cta_id = cluster_ctarank(),
            .warpgroup_id = warpgroup::groupid(),
            .warp_id = warpgroup::warpid(),
            .lane_id = warp::laneid(),
    
            .row_block_idx = 0,
            .col_block_idx = 0,
            .red_block_start = 0,
            .red_block_end = 0,

            .input_tiles = input_tiles,
            .input_scales = input_scales,
            .outputs = outputs,

            .tm_addr = tm_addr_shared,

            .tiles_arrived = tiles_arrived,
            .scales_sm_arrived = scales_sm_arrived,
            .scales_tm_arrived = scales_tm_arrived,
            .matmul_finished = matmul_finished,
            .tensor_finished = tensor_finished,
            .outputs_arrived = outputs_arrived,

            .stage = 0,
            .pending_stages = 0x0000'0000,
            .phasebits = 0xFFFF'0000,
        };

        // Execute the pipeline
        main_loop_policy::run(G, S);

        // Pipeline epilogue
        if (S.warpgroup_id == config::NUM_CONSUMERS && S.warp_id == TILE_LOADER && S.lane_id == 0) {
            #pragma unroll
            for (int i = 0; i < PIPELINE_STAGES; i++) {
                if ((S.pending_stages >> S.stage) & 0b1) {
                    tma::cluster::wait(S.matmul_finished[S.stage], get_phasebit<1>(S.phasebits, S.stage));
                    arrive(S.outputs_arrived);
                }
                S.stage = (S.stage + 1) % PIPELINE_STAGES;
            }
        }

        // De-allocate tensor memory
        asm volatile("{barrier.cluster.arrive.release.aligned;}");
        asm volatile("{barrier.cluster.wait.acquire.aligned;}");
        if (warpid() == 0) {
            asm volatile("{tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;}"
                :: "r"(S.tm_addr), "n"(512));
        }
    }

    template <typename main_loop_policy, typename globals>
    __device__ static inline void entrypoint(const globals &G) {
        if (warpgroup::groupid() == config::NUM_CONSUMERS) {
            warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
            dispatcher<main_loop_policy>(G);
        } else {
            warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
            dispatcher<main_loop_policy>(G);
        }
    }
};

using mp = mxfp8_matmul_pipeline;

struct globals {
    using A_gl = gl<fp8e4m3, 1, 1, -1, -1, mp::A_tile>; // M x K
    using B_gl = gl<fp8e4m3, 1, 1, -1, -1, mp::B_tile>; // N x K
    using C_gl = gl<bf16,    1, 1, -1, -1, mp::C_tile>; // M x N
    using A_sc_gl = gl<fp8e8m0, 1, -1, -1, mp::SCALE_BLOCK, mp::A_sc_vec>; // (M / 128) x (K / 128) x 512
    using B_sc_gl = gl<fp8e8m0, 1, -1, -1, mp::SCALE_BLOCK, mp::B_sc_vec>; // (N / 128) x (K / 128) x 512

    A_gl A;
    A_sc_gl A_sc;
    B_gl B;
    B_sc_gl B_sc;
    C_gl C;

    __host__ inline dim3 grid() { return dim3(mp::config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(mp::config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return mp::config::DYNAMIC_SHARED_MEMORY; }
};;

struct main_loop_policy {
    __device__ static inline void run(const globals &G, mp::state &S) {
        const int col_has_tail = (G.C.cols() / (mp::COL_BLOCK / 2)) & 0b1;
        const int row_has_tail = (G.C.rows() / (mp::ROW_BLOCK / 2)) & 0b1;
        const int num_blocks_per_row = G.C.cols() / mp::COL_BLOCK + col_has_tail;
        const int num_blocks_per_col = G.C.rows() / mp::ROW_BLOCK + row_has_tail;
        const int num_blocks = num_blocks_per_row * num_blocks_per_col;
        const int num_blocks_per_supergroup = mp::SUPERGROUP_SIZE * num_blocks_per_row;

        S.red_block_start = 0;
        S.red_block_end = G.A.cols() / mp::RED_BLOCK;

        for (int block_idx = clusterIdx().x; block_idx < num_blocks; block_idx += gridDim.x / mp::config::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(mp::SUPERGROUP_SIZE, num_blocks_per_col - supergroup_idx * mp::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;

            S.row_block_idx = supergroup_idx * mp::SUPERGROUP_SIZE + row_within_supergroup;
            S.col_block_idx = idx_within_supergroup / rows_in_supergroup;

            const bool is_tail_row = S.row_block_idx == (num_blocks_per_col - 1) && row_has_tail;
            const bool is_tail_col = S.col_block_idx == (num_blocks_per_row - 1) && col_has_tail;
            const bool load_A = !is_tail_row || S.cta_id == 0;
            const bool load_B = !is_tail_col || S.cta_id == 0;
            const bool do_load_store = load_A;
            const bool load_store_BC1 = !is_tail_col;

            if (S.warpgroup_id == mp::config::NUM_CONSUMERS && S.lane_id == 0) {
                if (S.warp_id == mp::TILE_LOADER) {
                    if (load_A && load_B)
                        mp::tile_loader_loop<true, true>(G, S);
                    else if (load_A)
                        mp::tile_loader_loop<true, false>(G, S);
                    else if (load_B)
                        mp::tile_loader_loop<false, true>(G, S);
                    else
                        mp::tile_loader_loop<false, false>(G, S);
                } else if (S.warp_id == mp::SCALE_SM_LOADER) {
                    if (do_load_store && load_store_BC1)
                        mp::scale_sm_loader_loop<true, true>(G, S);
                    else if (do_load_store)
                        mp::scale_sm_loader_loop<true, false>(G, S);
                    else
                        mp::scale_sm_loader_loop<false, false>(G, S);
                } else if (S.warp_id == mp::SCALE_TM_LOADER) {
                    mp::scale_tm_loader_loop(G, S);
                } else if (S.cta_id == 0 && S.warp_id == mp::MMA_LAUNCHER) {
                    mp::mma_launcher_loop(G, S);
                }
            } else if (S.warpgroup_id < mp::config::NUM_CONSUMERS) {
                if (do_load_store && load_store_BC1)
                    mp::consumer_loop<true, true>(G, S);
                else if (do_load_store)
                    mp::consumer_loop<true, false>(G, S);
                else
                    mp::consumer_loop<false, false>(G, S);
            }
        }
    }
};

__global__ __launch_bounds__(mp::config::NUM_THREADS, 1)
__cluster_dims__(mp::config::CLUSTER_SIZE)
void kernel(const __grid_constant__ globals G) {
    mp::entrypoint<main_loop_policy>(G);
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
