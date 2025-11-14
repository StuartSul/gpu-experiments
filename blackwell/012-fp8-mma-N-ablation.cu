/*
    Staying within 1-CTA matmuls, we get:
    - M=128 K=128 N=128 : 2130 TFLOPs
    - M=128 K=128 N=256 : 2600 TFLOPs

    Thus, we need to get N=256 matmuls running, but still support N that are only multiples of 128.

    The trick is:
    - We can declare a tail tile for B and C that are half the size of the main tile
        - Pass 2 TMA tiles for B and C
        - Alias B_tile and C_tile to B_tile_tail and C_tile_tail
        - Add boolean has_tail
        - Do one more computation with tail blocks if has_tail is true
    
    By this trick, we can do
    - M=128m K=128k N=128n : 2550 TFLOPs for N%256==0 inputs
    - M=128m K=128k N=128n : 2548 TFLOPs for N%256!=0 inputs

    There's an unknown bug where we need to add __syncthreads() after 
    has_stall declaration or else the kernel becomes literally 100x slower

    WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING
    ==> This code does NOT produce correct results. But this code is abandoned 
        because it is better to increase M rather than N (both in code simplicity and performance)
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

    static constexpr int NUM_WARPGROUPS = 3;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;

    static constexpr int PIPELINE_STAGES = 3;
};

// Kernel globals
struct globals {
    static constexpr int SUPERGROUP_BLOCKS = 12;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int COL_BLOCK_TAIL = COL_BLOCK / 2;
    static constexpr int REDUCTION_BLOCK = 128;

    using A_tile = st_fp8e4m3<ROW_BLOCK, REDUCTION_BLOCK>;
    using B_tile = st_fp8e4m3<COL_BLOCK, REDUCTION_BLOCK>;
    using B_tile_tail = st_fp8e4m3<COL_BLOCK_TAIL, REDUCTION_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 8, COL_BLOCK>;
    using C_tile_tail = st_bf<ROW_BLOCK / 8, COL_BLOCK_TAIL>;

    gl<fp8e4m3, 1, 1, -1, -1, A_tile> A;
    gl<fp8e4m3, 1, 1, -1, -1, B_tile, B_tile_tail> B;
    gl<bf16, 1, 1, -1, -1, C_tile, C_tile_tail> C;

    __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

// Kernel implementation
__global__ __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Warpgroup configuration
    using consumer = group<8>;
    int warpgroup_id = warpgroup::groupid();
    int warp_id = warpgroup::warpid();
    int lane_id = warp::laneid();

    // Allocate shared and tensor memory
    static_assert(sizeof(globals::A_tile) * config::PIPELINE_STAGES + 
                  sizeof(globals::B_tile) * config::PIPELINE_STAGES + 
                  sizeof(globals::C_tile) * config::PIPELINE_STAGES <= config::DYNAMIC_SHARED_MEMORY);
    globals::A_tile (&A_tile)[config::PIPELINE_STAGES] = allocator.allocate<globals::A_tile, config::PIPELINE_STAGES>();
    globals::B_tile (&B_tile)[config::PIPELINE_STAGES] = allocator.allocate<globals::B_tile, config::PIPELINE_STAGES>();
    globals::C_tile (&C_tile) = allocator.allocate<globals::C_tile>();
    tensor_allocator<1, 1> tm_allocator {};

    // This is okay because we know B/C tail tile sizes are multiples of 1024
    globals::B_tile_tail (&B_tile_tail)[config::PIPELINE_STAGES] = *reinterpret_cast<globals::B_tile_tail(*)[config::PIPELINE_STAGES]>(&B_tile);
    globals::C_tile_tail &C_tile_tail = *reinterpret_cast<globals::C_tile_tail *>(&C_tile);

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[config::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[config::PIPELINE_STAGES];
    __shared__ semaphore tensors_finished;
    __shared__ semaphore outputs_arrived;
    if (threadIdx.x == 0) {
        for (int i = 0; i < config::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(tensors_finished, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
    }
    __syncthreads();

    // Pipeline configuration
    int has_tail = G.C.cols() % globals::COL_BLOCK == globals::COL_BLOCK_TAIL;
    __syncthreads(); // unknown bug: without this, code is approximately 100x slower
    int num_blocks_per_row = G.C.cols() / globals::COL_BLOCK + has_tail;
    int num_blocks_per_col = G.C.rows() / globals::ROW_BLOCK;
    int num_blocks = num_blocks_per_row * num_blocks_per_col;
    int num_iters_per_block = G.A.cols() / globals::REDUCTION_BLOCK;
    int num_blocks_per_supergroup = globals::SUPERGROUP_BLOCKS * num_blocks_per_row;

    // Declare stage and phasebits for semaphore waits
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup_id == 2) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        // Sub divergence
        if (lane_id == 0) {
            int last_stage = -1;
            for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
                // Compute block indices
                int supergroup_idx = block_idx / num_blocks_per_supergroup;
                int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
                int rows_in_supergroup = min(globals::SUPERGROUP_BLOCKS, num_blocks_per_col - supergroup_idx * globals::SUPERGROUP_BLOCKS);
                int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
                int row_block_idx = supergroup_idx * globals::SUPERGROUP_BLOCKS + row_within_supergroup;
                int col_block_idx = idx_within_supergroup / rows_in_supergroup;

                // Main loop
                if (warp_id == 3) {
                    // Input loaders
                    if (!has_tail || col_block_idx < num_blocks_per_row - 1) {
                        for (int i = 0; i < num_iters_per_block; ++i) {
                            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                            if (stage == last_stage) {
                                arrive(outputs_arrived);
                                last_stage = -1;
                            }
                            tma::expect_bytes(inputs_arrived[stage], sizeof(globals::A_tile) + sizeof(globals::B_tile));
                            tma::load_async(A_tile[stage], G.A, {row_block_idx, i}, inputs_arrived[stage]);
                            tma::load_async(B_tile[stage], G.B, {col_block_idx, i}, inputs_arrived[stage]);
                            update_phasebit<1>(phasebits, stage);
                            if (i == num_iters_per_block - 1) {
                                last_stage = stage;
                            }
                            stage = (stage + 1) % config::PIPELINE_STAGES;
                        }
                    } else {
                        for (int i = 0; i < num_iters_per_block; ++i) {
                            wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                            if (stage == last_stage) {
                                arrive(outputs_arrived);
                                last_stage = -1;
                            }
                            tma::expect_bytes(inputs_arrived[stage], sizeof(globals::A_tile) + sizeof(globals::B_tile_tail));
                            tma::load_async(A_tile[stage], G.A, {row_block_idx, i}, inputs_arrived[stage]);
                            tma::load_async(B_tile_tail[stage], G.B, {col_block_idx * 2, i}, inputs_arrived[stage]);
                            update_phasebit<1>(phasebits, stage);
                            if (i == num_iters_per_block - 1) {
                                last_stage = stage;
                            }
                            stage = (stage + 1) % config::PIPELINE_STAGES;
                        }
                    }
                } else if (warp_id == 0) {
                    // TC launchers
                    if (!has_tail || col_block_idx < num_blocks_per_row - 1) {
                        using tm_t = tt<float, globals::ROW_BLOCK, globals::COL_BLOCK>;
                        tm_t tm = tm_allocator.template allocate<tm_t>(0);
                        wait(tensors_finished, get_phasebit<1>(phasebits, config::PIPELINE_STAGES));
                        update_phasebit<1>(phasebits, config::PIPELINE_STAGES);
                        {
                            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                            update_phasebit<0>(phasebits, stage);
                            mm_ABt(tm, A_tile[stage], B_tile[stage], inputs_finished[stage]);
                            stage = (stage + 1) % config::PIPELINE_STAGES;
                        }
                        for (int i = 1; i < num_iters_per_block; ++i) {
                            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage)); 
                            update_phasebit<0>(phasebits, stage);
                            mma_ABt(tm, A_tile[stage], B_tile[stage], inputs_finished[stage]);
                            stage = (stage + 1) % config::PIPELINE_STAGES;
                        }
                    } else {
                        using tm_t = tt<float, globals::ROW_BLOCK, globals::COL_BLOCK_TAIL>;
                        tm_t tm = tm_allocator.template allocate<tm_t>(0);
                        wait(tensors_finished, get_phasebit<1>(phasebits, config::PIPELINE_STAGES));
                        update_phasebit<1>(phasebits, config::PIPELINE_STAGES);
                        {
                            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                            update_phasebit<0>(phasebits, stage);
                            mm_ABt(tm, A_tile[stage], B_tile_tail[stage], inputs_finished[stage]);
                            stage = (stage + 1) % config::PIPELINE_STAGES;
                        }
                        for (int i = 1; i < num_iters_per_block; ++i) {
                            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage)); 
                            update_phasebit<0>(phasebits, stage);
                            mma_ABt(tm, A_tile[stage], B_tile_tail[stage], inputs_finished[stage]);
                            stage = (stage + 1) % config::PIPELINE_STAGES;
                        }
                    }
                }
            }
            if (warp_id == 3) {
                wait(inputs_finished[last_stage], get_phasebit<1>(phasebits, last_stage));
                arrive(outputs_arrived);
            }
        }
    } else {
        // Consumer group
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

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

            if (!has_tail || col_block_idx < num_blocks_per_row - 1) {
                using tm_t = tt<float, globals::ROW_BLOCK, globals::COL_BLOCK>;
                tm_t tm = tm_allocator.template allocate<tm_t>(0);
    
                // Load the output from tensor memory into registers
                rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK> C_reg;
                consumer::load_async(C_reg, tm);
                tensor_load_wait();
                consumer::sync(0);
                consumer::arrive(tensors_finished);
    
                // Store to global memory
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (warp_id * 2 + warpgroup_id == i) {
                        warp::store(C_tile, C_reg);
                        __syncwarp();
                        if (lane_id == 0) {
                            tma::store_async(G.C, C_tile, {row_block_idx * 8 + i, col_block_idx});
                            tma::store_async_read_wait();
                        }
                    }
                    consumer::sync(1);
                }
            } else {
                using tm_t = tt<float, globals::ROW_BLOCK, globals::COL_BLOCK_TAIL>;
                tm_t tm = tm_allocator.template allocate<tm_t>(0);
    
                // Load the output from tensor memory into registers
                rt_fl<globals::ROW_BLOCK / 8, globals::COL_BLOCK_TAIL> C_reg;
                consumer::load_async(C_reg, tm);
                tensor_load_wait();
                consumer::sync(0);
                consumer::arrive(tensors_finished);
    
                // Store to global memory
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (warp_id * 2 + warpgroup_id == i) {
                        warp::store(C_tile_tail, C_reg);
                        __syncwarp();
                        if (lane_id == 0) {
                            tma::store_async(G.C, C_tile_tail, {row_block_idx * 8 + i, col_block_idx});
                            tma::store_async_read_wait();
                        }
                    }
                    consumer::sync(1);
                }
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
