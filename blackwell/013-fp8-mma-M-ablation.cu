/*
    As stated in the previous test:
    Staying within 1-CTA matmuls, we get:
    - M=128 K=128 N=128 : 2130 TFLOPs
    - M=128 K=128 N=256 : 2600 TFLOPs

    Now, we can't go to M=256 N=256, because we run out of TM space (128x512) to store the scales
    Thus, another alternative is to set N=128 and increase the M.

    This actually makes it easier to cope with non-M%256==0 inputs, because we can just
    "turn off" subset of the tensor core for the tail iterations.

    Also, we can extend and run M=384 N=128 matmuls, using 3/4 of TM for matmuls and 1/4 for scales.

    Observation:
        - It's FASTER to increase M than to increase N
        - Pure M=256 K=128 N=128 : 2640 TFLOPs
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

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
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 128;
    static constexpr int REDUCTION_BLOCK = 128;

    using A_tile = st_fp8e4m3<ROW_BLOCK / 2, REDUCTION_BLOCK>;
    using B_tile = st_fp8e4m3<COL_BLOCK, REDUCTION_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 8, COL_BLOCK>;

    gl<fp8e4m3, 1, 1, -1, -1, A_tile> A;
    gl<fp8e4m3, 1, 1, -1, -1, B_tile> B;
    gl<bf16, 1, 1, -1, -1, C_tile> C;

    __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }

    struct pipeline_inputs {
        A_tile A[2];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };
};

// Kernel implementation
__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Warpgroup configuration
    using consumer = group<config::CONSUMER_WARPGROUPS * WARPGROUP_WARPS>;
    int warpgroup_id = warpgroup::groupid();
    int warp_id = warpgroup::warpid();
    int lane_id = warp::laneid();

    // Allocate shared and tensor memory
    static_assert(sizeof(globals::pipeline_inputs) * config::PIPELINE_STAGES + sizeof(globals::pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_inputs (&inputs)[config::PIPELINE_STAGES] = allocator.allocate<globals::pipeline_inputs, config::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs = allocator.allocate<globals::pipeline_outputs>();
    tensor_allocator<1, 1> tm_allocator {};

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[config::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[config::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore tensors_finished;
    if (threadIdx.x == 0) {
        for (int i = 0; i < config::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 2);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(tensors_finished, 0, 1);
    }
    __syncthreads();

    // Pipeline configuration
    int num_blocks_per_row = G.C.cols() / globals::COL_BLOCK;
    int num_blocks_per_col = G.C.rows() / globals::ROW_BLOCK;
    int num_blocks = num_blocks_per_row * num_blocks_per_col;
    int num_iters_per_block = G.A.cols() / globals::REDUCTION_BLOCK;
    int num_blocks_per_supergroup = globals::SUPERGROUP_BLOCKS * num_blocks_per_row;

    // Declare stage and phasebits for semaphore waits
    int stage = 0;
    int last_stage = -1;
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        // Sub divergence
        if (warp_id == 3 && lane_id == 0) {
            // Producer group -- loaders
            for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
                // Compute block indices
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(globals::SUPERGROUP_BLOCKS, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_BLOCKS);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * globals::SUPERGROUP_BLOCKS + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    if (stage == last_stage) {
                        arrive(outputs_arrived);
                        last_stage = -1;
                    }
                    tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                    tma::load_async(inputs[stage].A[0], G.A, {row_block_idx * 2 + 0, i}, inputs_arrived[stage]);
                    tma::load_async(inputs[stage].A[1], G.A, {row_block_idx * 2 + 1, i}, inputs_arrived[stage]);
                    tma::load_async(inputs[stage].B, G.B, {col_block_idx, i}, inputs_arrived[stage]);
                    update_phasebit<1>(phasebits, stage);
                    if (i == num_iters_per_block - 1) {
                        last_stage = stage;
                    }
                    stage = (stage + 1) % config::PIPELINE_STAGES;
                }
            }
            wait(inputs_finished[last_stage], get_phasebit<1>(phasebits, last_stage));
            arrive(outputs_arrived);
        } else if (warp_id == 0 && lane_id < 2) {
            // Producer group -- launchers
            using tm_t = tt<float, globals::ROW_BLOCK / 2, globals::COL_BLOCK>;
            tm_t tm = tm_allocator.template allocate<tm_t>(128 * lane_id);
            for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
                wait(tensors_finished, get_phasebit<1>(phasebits, config::PIPELINE_STAGES));
                update_phasebit<1>(phasebits, config::PIPELINE_STAGES);
                wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                update_phasebit<0>(phasebits, stage);
                mm_ABt(tm, inputs[stage].A[lane_id], inputs[stage].B, inputs_finished[stage]);
                stage = (stage + 1) % config::PIPELINE_STAGES;
                for (int i = 1; i < num_iters_per_block; ++i) {
                    wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage)); 
                    update_phasebit<0>(phasebits, stage);
                    mma_ABt(tm, inputs[stage].A[lane_id], inputs[stage].B, inputs_finished[stage]);
                    stage = (stage + 1) % config::PIPELINE_STAGES;
                }
            }
        }
    } else if (warpgroup_id < config::CONSUMER_WARPGROUPS) {
        // Consumer group
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        using tm_t = tt<float, globals::ROW_BLOCK / 2, globals::COL_BLOCK>;
        tm_t tm = tm_allocator.template allocate<tm_t>(warpgroup_id * 128);

        for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
            // Compute block indices
            int supergroup_idx = block_idx / num_blocks_per_supergroup;
            int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
            int rows_in_supergroup = min(globals::SUPERGROUP_BLOCKS, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_BLOCKS);
            int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
            int row_block_idx = supergroup_idx * globals::SUPERGROUP_BLOCKS + row_within_supergroup;
            int col_block_idx = idx_within_supergroup / rows_in_supergroup;

            // Wait for the last matmul to complete
            wait(outputs_arrived, get_phasebit<0>(phasebits, config::PIPELINE_STAGES));
            update_phasebit<0>(phasebits, config::PIPELINE_STAGES);

            // Load the output from tensor memory into registers
            rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_reg;
            warpgroup::load_async(C_reg, tm);
            tensor_load_wait();
            consumer::sync(1); // wait for both consumer WGs, to arrive asap
            if (consumer::laneid() == 0)
                arrive(tensors_finished);

            // Store to global memory
            for (int i = 0; i < 8; ++i) {
                if (consumer::warpid() == i) {
                    warp::store(outputs.C, C_reg);
                    __syncwarp();
                    if (lane_id == 0) {
                        tma::store_async(G.C, outputs.C, {row_block_idx * 8 + i, col_block_idx});
                        tma::store_async_read_wait();
                    }
                }
                consumer::sync(1);
            }
        }
    }
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
