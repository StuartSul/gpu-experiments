/*
    2-CTA FP8 Matrix Multiplication

    Benchmarks:

        Pure C++ (Thunderkittens/kernels/matmul/FP8_B200):
            - 4096x4096x4096 : 2107.6 TFLOPs
            - 8192x8192x8192 : 3279.99 TFLOPs
            - 16384x16384x16384 : 3290.66 TFLOPs
            - 204800x2048x1408 : 3390.25 TFLOPs (but results incorrect, so unsure if accurate measure)

        ^ + called from PyTorch (this file)
            - 4096x4096x4096 : 2090.42 TFLOp/s
            - 8192x8192x8192 : 3200.70 TFLOp/s
            - 16384x16384x16384 : 3300.45 TFLOp/s
            - 204800x2048x1536 : 2887.71 TFLOp/s (changed dim for yet supported granularity)
        
        ^ + WITH L2 cache clear (this file)
            - 4096x4096x4096 : 1231.25 TFLOp/s
            - 8192x8192x8192 : 2836.37 TFLOp/s
            - 16384x16384x16384 : 3202.16 TFLOp/s
            - 204800x2048x1536 : 2707.09 TFLOp/s

        ^ + pipeline factored out
        ^ + doing 256x128x256 instead of 512x128x256
        ^ + supporting 128 granularity on M
        ^ + supporting 128 granularity on N
        ^ + supporting 128 granularity on K
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

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

    static constexpr int PRODUCER_REGISTERS = 72;
    static constexpr int CONSUMER_REGISTERS = 216;
};

// Kernel globals
struct globals {
    static constexpr int PIPELINE_STAGES = 4;

    // Per-block dimensions
    static constexpr int ROW_BLOCK = 512;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 128; // reduction axis

    // Supergrouping for higher L2 utilization
    static constexpr int SUPERGROUP_SIZE = 8;

    using A_tile = st_fp8e4m3<ROW_BLOCK / 4, RED_BLOCK>; // WG/CTA distributed
    using B_tile = st_fp8e4m3<COL_BLOCK / 2, RED_BLOCK>; // CTA distributed
    using C_tile = st_bf<ROW_BLOCK / 4, COL_BLOCK / 4>; // WG/CTA distributed + array-divided

    using A_gl = gl<fp8e4m3, 1, 1, -1, -1, A_tile>;
    using B_gl = gl<fp8e4m3, 1, 1, -1, -1, B_tile>;
    using C_gl = gl<bf16,    1, 1, -1, -1, C_tile>;

    A_gl A;
    B_gl B;
    C_gl C;

    __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }

    struct pipeline_inputs {
        A_tile A[config::NUM_CONSUMERS];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };
};

// Kernel implementation
__global__ __launch_bounds__(config::NUM_THREADS, 1) 
__cluster_dims__(config::CLUSTER_SIZE)
void kernel(const __grid_constant__ globals G) {
    // Declare shared memory
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);

    // Allocate shared memory
    static_assert(sizeof(globals::pipeline_inputs) * globals::PIPELINE_STAGES + sizeof(globals::pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_inputs (&inputs)[globals::PIPELINE_STAGES] = sm_allocator.allocate<globals::pipeline_inputs, globals::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs = sm_allocator.allocate<globals::pipeline_outputs>();

    // Allocate tensor memory
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator {};
    using tm_t = tt<float, globals::ROW_BLOCK / 4, globals::COL_BLOCK>;

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[globals::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[globals::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore outputs_finished[config::NUM_CONSUMERS];
    if (threadIdx.x == 32) {
        #pragma unroll
        for (int i = 0; i < globals::PIPELINE_STAGES; i++) {
            init_semaphore(inputs_arrived[i], 0, config::CLUSTER_SIZE);
            init_semaphore(inputs_finished[i], 0, config::NUM_CONSUMERS);
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < config::NUM_CONSUMERS; i++) {
            init_semaphore(outputs_finished[i], 0, config::CLUSTER_SIZE);
        }
    }
    everyone::tma::cluster::sync();

    // Set up pipeline parameters
    uint32_t stage = 0;
    uint32_t last_stage = globals::PIPELINE_STAGES;
    uint32_t phasebits = 0xFFFF'0000;
    const int lane_id = warp::laneid();
    const int warp_id = warpgroup::warpid();
    const int warpgroup_id = warpgroup::groupid();
    const int cta_id = cluster_ctarank();
    const int cluster_id = clusterIdx().x;
    const int num_blocks_per_row = G.C.cols() / globals::COL_BLOCK;
    const int num_blocks_per_col = G.C.rows() / globals::ROW_BLOCK;
    const int num_blocks = num_blocks_per_row * num_blocks_per_col;
    const int num_iters = G.A.cols() / globals::RED_BLOCK;
    const int num_blocks_per_supergroup = globals::SUPERGROUP_SIZE * num_blocks_per_row;
    using consumer = group<WARPGROUP_WARPS * config::NUM_CONSUMERS>;

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 3 && lane_id == 0) {
            // Load input matrices
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(globals::SUPERGROUP_SIZE, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_SIZE);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * globals::SUPERGROUP_SIZE + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;
                
                for (int red_block_idx = 0; red_block_idx < num_iters; red_block_idx++) {
                    tma::cluster::wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
    
                    if (stage == last_stage) {
                        arrive(outputs_arrived);
                        last_stage = globals::PIPELINE_STAGES;
                    }

                    // Load to current CTA, but signal mbarrier at CTA 0 (tma::cluster is purely for cluster-level synchronization)
                    tma::cluster::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs), 0);
                    warp::tma::cluster::load_async(inputs[stage].A[0], G.A, {row_block_idx * 4 + cta_id * 2 + 0, red_block_idx}, inputs_arrived[stage], (uint16_t)(1 << cta_id), 0); 
                    warp::tma::cluster::load_async(inputs[stage].A[1], G.A, {row_block_idx * 4 + cta_id * 2 + 1, red_block_idx}, inputs_arrived[stage], (uint16_t)(1 << cta_id), 0);
                    warp::tma::cluster::load_async(inputs[stage].B,    G.B, {col_block_idx * 2 + cta_id,         red_block_idx}, inputs_arrived[stage], (uint16_t)(1 << cta_id), 0);
    
                    if (red_block_idx == num_iters - 1) {
                        last_stage = stage;
                    }

                    // Update stage
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
            if (last_stage < globals::PIPELINE_STAGES) {
                tma::cluster::wait(inputs_finished[last_stage], get_phasebit<1>(phasebits, last_stage));
                arrive(outputs_arrived);
            }
        } else if (cta_id == 0 && warp_id == 0 && lane_id <= 1) {
            // Launch tensor core matrix multiply
            tm_t tm = tm_allocator.allocate<tm_t>(lane_id * globals::COL_BLOCK);

            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {                
                for (int red_block_idx = 0; red_block_idx < num_iters; red_block_idx++) {
                    if (red_block_idx == 0) {
                        tma::cluster::wait(outputs_finished[lane_id], get_phasebit<1>(phasebits, globals::PIPELINE_STAGES));
                        update_phasebit<1>(phasebits, globals::PIPELINE_STAGES);
                    }
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    if (red_block_idx == 0)
                        mm2_ABt(tm, inputs[stage].A[lane_id], inputs[stage].B, inputs_finished[stage]);
                    else
                        mma2_ABt(tm, inputs[stage].A[lane_id], inputs[stage].B, inputs_finished[stage]);
                    stage = (stage + 1) % globals::PIPELINE_STAGES;
                }
            }
        }
    } else {
        // Consumer group
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        tm_t tm = tm_allocator.allocate<tm_t>(warpgroup_id * globals::COL_BLOCK);

        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / config::CLUSTER_SIZE) {
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(globals::SUPERGROUP_SIZE, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_SIZE);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * globals::SUPERGROUP_SIZE + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            // Wait for the matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, globals::PIPELINE_STAGES));
            update_phasebit<0>(phasebits, globals::PIPELINE_STAGES);

            // Load the output from tensor memory into registers
            rt_bf<globals::ROW_BLOCK / 16, globals::COL_BLOCK / 4> C_reg[4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                warpgroup::load_async(C_reg[i], tm.subtile<tt<float, globals::ROW_BLOCK / 4, globals::COL_BLOCK / 4>>(0, i * globals::COL_BLOCK / 4));
            tensor_load_wait();
            warpgroup::sync(2 + warpgroup_id);
            if (warpgroup::laneid() == 0)
                tma::cluster::arrive(outputs_finished[warpgroup_id], 0, 1); // signal CTA 0

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                if (warpgroup::groupid() == i) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        warpgroup::store(outputs.C, C_reg[j]);
                        warpgroup::sync(2 + warpgroup_id);
                        if (warpgroup::laneid() == 0) {
                            tma::store_async(G.C, outputs.C, {row_block_idx * 4 + cta_id * 2 + i, col_block_idx * 4 + j});
                            tma::store_async_read_wait();
                        }
                        warpgroup::sync(2 + warpgroup_id);
                    }
                }
                consumer::sync(1);
            }
        }
    }

    everyone::tma::cluster::sync(); // for tm_allocator destructor
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel>(m, "kernel",
        &globals::A,
        &globals::B,
        &globals::C
    );
}
