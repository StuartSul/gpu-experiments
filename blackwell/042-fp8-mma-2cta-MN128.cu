/*
    2-CTA FP8 Matrix Multiplication, refactored.
    The refactored version allows easily creating different variants of matmuls, by inheriting the base class
    and overriding one or two functions. Specifically, it makes it easier to change which row/col/reduction range
    is handled at each phase.

    Benchmarks so far:

        Pure C++ (Thunderkittens/kernels/matmul/FP8_B200):
            - 4096x4096x4096 : 2107.6 TFLOPs
            - 8192x8192x8192 : 3279.99 TFLOPs
            - 16384x16384x16384 : 3290.66 TFLOPs
            - 204800x2048x1408 : 3390.25 TFLOPs (but results incorrect, so unsure if accurate measure)

        ^ + called from PyTorch
            - 4096x4096x4096 : 2090.42 TFLOp/s
            - 8192x8192x8192 : 3200.70 TFLOp/s
            - 16384x16384x16384 : 3300.45 TFLOp/s
            - 204800x2048x1536 : 2887.71 TFLOp/s (changed dim for yet supported granularity)
        
        ^ + WITH L2 cache clear
            - 4096x4096x4096 : 1231.25 TFLOp/s
            - 8192x8192x8192 : 2836.37 TFLOp/s
            - 16384x16384x16384 : 3202.16 TFLOp/s
            - 204800x2048x1536 : 2707.09 TFLOp/s

        ^ + pipeline factored out
            - 4096x4096x4096 : 1243.66 TFLOp/s
            - 8192x8192x8192 : 2900.05 TFLOp/s
            - 16384x16384x16384 : 3190.50 TFLOp/s
            - 204800x2048x1536 : 2652.84 TFLOp/s

        ^ + doing 256x128x256 instead of 512x128x256
          + with less smem usage, we can increase pipeline stages --> 5 stages perform the best (gets slow on 6)
            - 4096x4096x4096 : 1300.71 TFLOp/s
            - 8192x8192x8192 : 2763.66 TFLOp/s
            - 16384x16384x16384 : 3033.94 TFLOp/s
            - 204800x2048x1536 : 2490.36 TFLOp/s

        ^ + supporting 128 granularity on M & N (this file)
            - 4096x4096x4096 : 1281.16 TFLOp/s
            - 8192x8192x8192 : 2719.50 TFLOp/s
            - 16384x16384x16384 : 2912.10 TFLOp/s (degraded due to GPU. probably should add about 70 TFLOPs to this)
            - 204800x2048x1536 : 2486.30 TFLOp/s

        ^ + supporting 128 granularity on K
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct fp8_matmul_base {

    // Pipeline stages
    static constexpr int PIPELINE_STAGES = 5;

    // Per-block dimensions
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 128; // reduction axis

    // Supergrouping for higher L2 utilization
    static constexpr int SUPERGROUP_SIZE = 12;

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
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK / 2>;  // CTA distributed + WG divided
    using tm_t = tt<float, ROW_BLOCK / 2, COL_BLOCK>;
    using consumer = group<config::NUM_CONSUMERS * WARPGROUP_WARPS>;

    struct globals {
        using A_gl = gl<fp8e4m3, 1, 1, -1, -1, A_tile>;
        using B_gl = gl<fp8e4m3, 1, 1, -1, -1, B_tile>;
        using C_gl = gl<bf16,    1, 1, -1, -1, C_tile>;
    
        A_gl A;
        B_gl B;
        C_gl C;
    
        __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
        __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
        __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
    };

    struct pipeline_inputs {
        A_tile A;
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };

    // Pipeline state
    struct state {
        const bool is_input_loader; // 1 thread
        const bool is_launcher;     // 1 thread
        const bool is_consumer;     // 2 warpgroups

        const int cta_id;      // 0 or 1
        const int consumer_id; // 0 or 1 (valid if is_consumer)

        int row_block_idx;
        int col_block_idx;
        int red_block_start;
        int red_block_end;

        pipeline_inputs (&inputs)[PIPELINE_STAGES];
        pipeline_outputs &outputs;

        tensor_allocator<1, config::CLUSTER_SIZE> &tm_allocator;

        semaphore (&inputs_arrived)[PIPELINE_STAGES];
        semaphore (&inputs_finished)[PIPELINE_STAGES];
        semaphore &outputs_arrived;
        semaphore &outputs_finished;

        uint32_t stage;
        uint32_t last_stage;
        uint32_t phasebits;
    };

    template <bool load_A, bool load_B>
    __device__ static inline void input_loader_loop(const globals &G, state &S) {
        constexpr int bytes_per_iter = (load_A ? sizeof(A_tile) : 0) + (load_B ? sizeof(B_tile) : 0);

        for (int red_block_idx = S.red_block_start; red_block_idx < S.red_block_end; red_block_idx++) {
            tma::cluster::wait(S.inputs_finished[S.stage], get_phasebit<1>(S.phasebits, S.stage));
            update_phasebit<1>(S.phasebits, S.stage);

            if (S.stage == S.last_stage) {
                arrive(S.outputs_arrived);
                S.last_stage = PIPELINE_STAGES;
            }

            // Load to current CTA, but signal mbarrier at CTA 0 (tma::cluster is purely for cluster-level synchronization)
            if constexpr (load_A || load_B) {
                tma::cluster::expect_bytes(S.inputs_arrived[S.stage], bytes_per_iter, 0);
                if constexpr (load_A)
                    tma::cluster::load_async(S.inputs[S.stage].A, G.A, {S.row_block_idx * 2 + S.cta_id, red_block_idx}, S.inputs_arrived[S.stage], (uint16_t)(1 << S.cta_id), 0); 
                if constexpr (load_B)
                    tma::cluster::load_async(S.inputs[S.stage].B, G.B, {S.col_block_idx * 2 + S.cta_id, red_block_idx}, S.inputs_arrived[S.stage], (uint16_t)(1 << S.cta_id), 0);
            } else {
                tma::cluster::arrive(S.inputs_arrived[S.stage], 0, 1); // signal immediately
            }

            if (red_block_idx == S.red_block_end - 1) {
                S.last_stage = S.stage;
            }

            // Update stage
            S.stage = (S.stage + 1) % PIPELINE_STAGES;
        }
    }

    __device__ static inline void launcher_loop(const globals &G, state &S) {
        tm_t tm = S.tm_allocator.allocate<tm_t>(0);
        for (int red_block_idx = S.red_block_start; red_block_idx < S.red_block_end; red_block_idx++) {
            if (red_block_idx == S.red_block_start) {
                tma::cluster::wait(S.outputs_finished, get_phasebit<1>(S.phasebits, PIPELINE_STAGES));
                update_phasebit<1>(S.phasebits, PIPELINE_STAGES);
            }
            tma::cluster::wait(S.inputs_arrived[S.stage], get_phasebit<0>(S.phasebits, S.stage));
            update_phasebit<0>(S.phasebits, S.stage);
            if (red_block_idx == S.red_block_start)
                mm2_ABt(tm, S.inputs[S.stage].A, S.inputs[S.stage].B, S.inputs_finished[S.stage]);
            else
                mma2_ABt(tm, S.inputs[S.stage].A, S.inputs[S.stage].B, S.inputs_finished[S.stage]);
            S.stage = (S.stage + 1) % PIPELINE_STAGES;
        }
    }

    template <bool save_output, bool save_full_col>
    __device__ static inline void consumer_loop(const globals &G, state &S) {
        tm_t tm = S.tm_allocator.allocate<tm_t>(0);

        // Wait for the matmul to complete
        wait(S.outputs_arrived, get_phasebit<0>(S.phasebits, PIPELINE_STAGES));
        update_phasebit<0>(S.phasebits, PIPELINE_STAGES);

        // Load the output from tensor memory into registers
        rt_bf<ROW_BLOCK / 8, COL_BLOCK / 2> C_reg;
        if (S.red_block_start >= S.red_block_end) {
            warp::zero(C_reg);
        } else {
            warpgroup::load_async(C_reg, tm.subtile<tt<float, ROW_BLOCK / 2, COL_BLOCK / 2>>(0, S.consumer_id * COL_BLOCK / 2));
            tensor_load_wait();
            consumer::sync(1);
            if (consumer::laneid() == 0)
                tma::cluster::arrive(S.outputs_finished, 0, 1); // signal CTA 0
        }

        constexpr int num_store_iters = save_output ? (save_full_col ? 2 : 1) : 0;

        #pragma unroll
        for (int i = 0; i < num_store_iters; i++) {
            if (S.consumer_id == i) {
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

    __device__ static inline void main_loop(const globals &G, state &S) {
        const int col_has_tail = (G.C.cols() / (COL_BLOCK / 2)) & 0b1;
        const int row_has_tail = (G.C.rows() / (ROW_BLOCK / 2)) & 0b1;
        const int num_blocks_per_row = G.C.cols() / COL_BLOCK + col_has_tail;
        const int num_blocks_per_col = G.C.rows() / ROW_BLOCK + row_has_tail;
        const int num_blocks = num_blocks_per_row * num_blocks_per_col;
        const int num_blocks_per_supergroup = SUPERGROUP_SIZE * num_blocks_per_row;

        S.red_block_start = 0;
        S.red_block_end = G.A.cols() / RED_BLOCK;

        for (int block_idx = clusterIdx().x; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(SUPERGROUP_SIZE, num_blocks_per_col - supergroup_idx * SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;

            S.row_block_idx = supergroup_idx * SUPERGROUP_SIZE + row_within_supergroup;
            S.col_block_idx = idx_within_supergroup / rows_in_supergroup;

            const bool is_tail_row = S.row_block_idx == (num_blocks_per_col - 1) && row_has_tail;
            const bool is_tail_col = S.col_block_idx == (num_blocks_per_row - 1) && col_has_tail;
            const bool load_A = !is_tail_row || S.cta_id == 0;
            const bool load_B = !is_tail_col || S.cta_id == 0;
            const bool save_output = load_A;
            const bool save_full_col = !is_tail_col;

            if (S.is_input_loader) {
                if (load_A && load_B)
                    input_loader_loop<true, true>(G, S);
                else if (load_A)
                    input_loader_loop<true, false>(G, S);
                else if (load_B)
                    input_loader_loop<false, true>(G, S);
                else
                    input_loader_loop<false, false>(G, S);
            } else if (S.is_launcher) {
                launcher_loop(G, S);
            } else if (S.is_consumer) {
                if (save_output && save_full_col)
                    consumer_loop<true, true>(G, S);
                else if (save_output)
                    consumer_loop<true, false>(G, S);
                else
                    consumer_loop<false, false>(G, S);
            }
        }
    }

    __device__ static inline void dispatcher(const globals &G) {
        // Declare shared memory
        extern __shared__ int __shm[]; 
        tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    
        // Allocate shared memory
        static_assert(sizeof(pipeline_inputs) * PIPELINE_STAGES + sizeof(pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY);
        pipeline_inputs (&inputs)[PIPELINE_STAGES] = sm_allocator.allocate<pipeline_inputs, PIPELINE_STAGES>();
        pipeline_outputs &outputs = sm_allocator.allocate<pipeline_outputs>();
    
        // Allocate tensor memory
        tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator {};
    
        // Declare mbarriers
        __shared__ semaphore inputs_arrived[PIPELINE_STAGES];
        __shared__ semaphore inputs_finished[PIPELINE_STAGES];
        __shared__ semaphore outputs_arrived;
        __shared__ semaphore outputs_finished;

        // Initialize mbarriers
        if (threadIdx.x == 32) {
            #pragma unroll
            for (int i = 0; i < PIPELINE_STAGES; i++) {
                init_semaphore(inputs_arrived[i], 0, config::CLUSTER_SIZE);
                init_semaphore(inputs_finished[i], 0, 1);
            }
            init_semaphore(outputs_arrived, 0, 1);
            init_semaphore(outputs_finished, 0, config::CLUSTER_SIZE);
        }
        everyone::tma::cluster::sync();

        // Get thread/block indices
        int lane_id = warp::laneid();
        int warp_id = warpgroup::warpid();
        int warpgroup_id = warpgroup::groupid();
        int cta_id = cluster_ctarank();

        // Set up matmul pipeline state
        state S {
            .is_input_loader = (warpgroup_id == config::NUM_CONSUMERS) && (warp_id == 3) && (lane_id == 0),
            .is_launcher = (warpgroup_id == config::NUM_CONSUMERS) && (cta_id == 0) && (warp_id == 0) && (lane_id == 0),
            .is_consumer = warpgroup_id < config::NUM_CONSUMERS,

            .cta_id = cta_id,
            .consumer_id = warpgroup_id,
    
            .row_block_idx = 0,
            .col_block_idx = 0,
            .red_block_start = 0,
            .red_block_end = 0,

            .inputs = inputs,
            .outputs = outputs,
            .tm_allocator = tm_allocator,

            .inputs_arrived = inputs_arrived,
            .inputs_finished = inputs_finished,
            .outputs_arrived = outputs_arrived,
            .outputs_finished = outputs_finished,

            .stage = 0,
            .last_stage = PIPELINE_STAGES,
            .phasebits = 0xFFFF'0000,
        };

        // Execute the pipeline
        main_loop(G, S);

        // Pipeline epilogue
        if (S.is_input_loader && S.last_stage < PIPELINE_STAGES) {
            tma::cluster::wait(S.inputs_finished[S.last_stage], get_phasebit<1>(S.phasebits, S.last_stage));
            arrive(S.outputs_arrived);
        }
        everyone::tma::cluster::sync(); // for tm_allocator destructor
    }

    __device__ static inline void entrypoint(const globals &G) {
        if (warpgroup::groupid() == config::NUM_CONSUMERS) {
            warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();
            dispatcher(G);
        } else {
            warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
            dispatcher(G);
        }
    }
};

using mp = fp8_matmul_base;

__global__ __launch_bounds__(mp::config::NUM_THREADS, 1) 
__cluster_dims__(mp::config::CLUSTER_SIZE)
void kernel(const __grid_constant__ mp::globals G) {
    mp::entrypoint(G);
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel>(m, "kernel",
        &mp::globals::A,
        &mp::globals::B,
        &mp::globals::C
    );
}
