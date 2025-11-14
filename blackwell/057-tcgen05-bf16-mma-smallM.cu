#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int SM_COUNT = 148;
    static constexpr int STATIC_SHARED_MEMORY = 128;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 56;
    static constexpr int CONSUMER_REGISTERS = 224;

    static constexpr int PIPELINE_STAGES = 4;
};

struct globals {
    static constexpr int SUPERGROUP_BLOCKS = 8;
    static constexpr int ROW_BLOCK = 64;
    static constexpr int COL_BLOCK = 128;
    static constexpr int REDUCTION_BLOCK = 128;

    using A_tile = st_bf<ROW_BLOCK, REDUCTION_BLOCK>;
    using B_tile = st_bf<COL_BLOCK, REDUCTION_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK, COL_BLOCK / 8>;

    gl<bf16, 1, 1, -1, -1, A_tile> A;
    gl<bf16, 1, 1, -1, -1, B_tile> B;
    gl<bf16, 1, 1, -1, -1, C_tile> C;

    __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }

    struct pipeline_inputs {
        A_tile A;
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };
};

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
    tensor_allocator<1, config::CLUSTER_SIZE> tm_allocator {};

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[config::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[config::PIPELINE_STAGES];
    __shared__ semaphore outputs_arrived;
    __shared__ semaphore tensors_finished;
    if (threadIdx.x == 0) {
        for (int i = 0; i < config::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
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
                    update_phasebit<1>(phasebits, stage);
                    if (stage == last_stage) {
                        arrive(outputs_arrived);
                        last_stage = -1;
                    }
                    tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                    tma::load_async(inputs[stage].A, G.A, {row_block_idx, i}, inputs_arrived[stage]);
                    tma::load_async(inputs[stage].B, G.B, {col_block_idx, i}, inputs_arrived[stage]);
                    if (i == num_iters_per_block - 1) {
                        last_stage = stage;
                    }
                    stage = (stage + 1) % config::PIPELINE_STAGES;
                }
            }
            if (last_stage >= 0) {
                wait(inputs_finished[last_stage], get_phasebit<1>(phasebits, last_stage));
                arrive(outputs_arrived);
            }
        } else if (warp_id == 0 && lane_id == 0) {
            // Producer group -- launchers
            auto tm = tm_allocator.allocate<tt<float, globals::ROW_BLOCK, globals::COL_BLOCK>>(0, 0);
            for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
                wait(tensors_finished, get_phasebit<1>(phasebits, config::PIPELINE_STAGES));
                update_phasebit<1>(phasebits, config::PIPELINE_STAGES);
                for (int i = 0; i < num_iters_per_block; ++i) {
                    wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    if (i == 0)
                        mm_ABt(tm, inputs[stage].A, inputs[stage].B, inputs_finished[stage]);
                    else
                        mma_ABt(tm, inputs[stage].A, inputs[stage].B, inputs_finished[stage]);
                    stage = (stage + 1) % config::PIPELINE_STAGES;
                }
            }
        }
    } else {
        // Consumer group
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        auto tm = tm_allocator.allocate<tt<float, globals::ROW_BLOCK, globals::COL_BLOCK>>(0, 0);

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
            rt_bf<globals::ROW_BLOCK / 4, globals::COL_BLOCK / 8> C[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                consumer::load_async(C[i], tm.subtile<tt<float, globals::ROW_BLOCK, globals::COL_BLOCK / 8>>(0, i * globals::COL_BLOCK / 8));
            tensor_load_wait();
            consumer::sync(1);
            if (consumer::laneid() == 0)
                arrive(tensors_finished, 0);

            // Store to global memory
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                consumer::store(outputs.C, C[i]);
                consumer::sync(1);
                consumer::tma::store_async(G.C, outputs.C, {row_block_idx, col_block_idx * 8 + i});
                consumer::tma::store_async_read_wait();
                consumer::sync(1);
            }
        }
    }
}

PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel>(m, "bf16_matmul",
        &globals::A,
        &globals::B,
        &globals::C
    );
}
