/*
    Obaservations:
    - This kernel reaches about 4725 GB/s
    - Without consumer, it reaches around 7377 GB/s. 
        - Theoretical max is ~ 8 TB/s. So this is good. No problem with the producer
    - Without consumer::store's from reg -> smem, it is 6979 GB/s.
        - The culprit is mostly the shared memory writes

    The above observations are wrong.

    Real benchmark:
        Non-transposed:
            - 4276.27 GB/s with consumer
            - 5109.70 GB/s without consumer
        Transposed:
            - 3875 GB/s with consumer
            - 4736 GB/s without consumer

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

    static constexpr int PIPELINE_STAGES = 6;
};

// Kernel globals
struct globals {
    // 1. The block size should be equivalent to the quantization block size
    // 2. The block should be square (for transpose)
    static constexpr int BLOCK_SIZE = 128;
    static constexpr int QUANT_BLOCK_SIZE = 32;
    static_assert(BLOCK_SIZE / QUANT_BLOCK_SIZE == 4);

    using A_tile_bf16 = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using A_tile_fp8 = st_fp8e4m3<BLOCK_SIZE, BLOCK_SIZE>;
    using A_sc_vec = sv<fp8e8m0, BLOCK_SIZE * BLOCK_SIZE / QUANT_BLOCK_SIZE>;

    gl<bf16, 1, -1, -1, -1, A_tile_bf16> A_bf16;
    gl<fp8e4m3, 1, -1, -1, -1, A_tile_fp8> A_fp8;
    gl<fp8e8m0, -1, -1, -1, -1, A_sc_vec> A_sc;

    __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }

    struct pipeline_inputs {
        A_tile_bf16 A_bf16;
    };

    struct pipeline_outputs {
        A_tile_fp8 A_fp8;
        A_sc_vec A_sc;
    };
};

// Load from shared memory to register tile, and transpose in the process
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void transpose_load(RT &dst, const ST &src) {
    constexpr int GROUP_WARPS = config::CONSUMER_WARPGROUPS * WARPGROUP_WARPS;
    constexpr int height = ST::height;
    constexpr int warp_height = RT::height;
    static_assert(ST::height == ST::width, "Transpose load requires square tiles.");
    static_assert(RT::height * GROUP_WARPS == RT::width, "Transpose load requires square tiles.");
    static_assert(std::is_same_v<typename RT::layout, ducks::rt_layout::row>, "Only row-major layout is supported.");
    static_assert((height * config::CONSUMER_WARPGROUPS) % GROUP_WARPS == 0, "Group load / store requires tile height to be a multiple of GROUP_WARPS.");
    static_assert(height % warp_height == 0, "Group load / store requires tile height to be a multiple of the RT height.");
    static_assert(ST::width == RT::width, "Group load / store requires tile widths to match.");
    static_assert(sizeof(typename ST::dtype) == 2, "Fix this function.");
    int local_warpid;
    if constexpr(GROUP_WARPS % 4 == 0) local_warpid = (warpid()/4+(warpid()%4)*(GROUP_WARPS/4));
    else local_warpid = warpid();
    using T2 = RT::dtype;
    using U  = ST::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = ::kittens::laneid();

    // convert to shared state space
    uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            U2 tmp[4];
            int row = j * dst.tile_size_col + (warp_laneid % 16);
            int col = (local_warpid * warp_height + i) * dst.tile_size_row + (warp_laneid / 16) * 8;
            move<U2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(shared_addr, {row % ST::rows, col}));
            dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
            dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
            dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(tmp[2]);
            dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(tmp[3]);
            warp::transpose_inplace(dst.tiles[i][j]);
        }
    }
}

// Kernel implementation
template <bool transpose>
__global__  __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Warpgroup configuration
    int lane_id = warp::laneid();
    int warp_id = warpgroup::warpid();
    int warpgroup_id = warpgroup::groupid();
    using consumer = group<config::CONSUMER_WARPGROUPS * WARPGROUP_WARPS>;

    // Allocate shared memory
    static_assert(sizeof(globals::pipeline_inputs) * config::PIPELINE_STAGES + sizeof(globals::pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_inputs (&inputs)[config::PIPELINE_STAGES] = allocator.allocate<globals::pipeline_inputs, config::PIPELINE_STAGES>();
    globals::pipeline_outputs &outputs = allocator.allocate<globals::pipeline_outputs>();

    // Set up mbarriers
    __shared__ semaphore inputs_arrived[config::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[config::PIPELINE_STAGES];
    if (threadIdx.x == 0) {
        for (int i = 0; i < config::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
    }
    __syncthreads();

    // Pipeline configuration
    int num_groups = G.A_bf16.depth();
    int num_blocks_per_row = G.A_bf16.cols() / globals::BLOCK_SIZE;
    int num_blocks_per_col = G.A_bf16.rows() / globals::BLOCK_SIZE;
    int num_blocks_per_group = num_blocks_per_row * num_blocks_per_col;
    int num_blocks = num_groups * num_blocks_per_group;

    // Declare phasebits for semaphore waits
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        // Main loop
        int stage = 0;
        if (warp_id == 0 && lane_id == 0) {
            for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
                // Compute block indices
                int group_idx = block_idx / num_blocks_per_group;
                int group_local_block_idx = block_idx % num_blocks_per_group;
                int row_block_idx = group_local_block_idx / num_blocks_per_row;
                int col_block_idx = group_local_block_idx % num_blocks_per_row;

                // Wait for shared memory to be free
                wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);

                // Load inputs into shared memory
                tma::expect_bytes(inputs_arrived[stage], sizeof(globals::pipeline_inputs));
                tma::load_async(inputs[stage].A_bf16, G.A_bf16, {group_idx, row_block_idx, col_block_idx}, inputs_arrived[stage]);

                // Update phasebit and stage
                stage = (stage + 1) % config::PIPELINE_STAGES;
            }
        }
    } else {
        // Consumer group
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        // Main loop
        int stage = 0;
        for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
            // Compute block indices
            int group_idx = block_idx / num_blocks_per_group;
            int group_local_block_idx = block_idx % num_blocks_per_group;
            int row_block_idx, col_block_idx;
            if constexpr (transpose) {
                row_block_idx = group_local_block_idx % num_blocks_per_row;
                col_block_idx = group_local_block_idx / num_blocks_per_row;
            } else {
                row_block_idx = group_local_block_idx / num_blocks_per_row;
                col_block_idx = group_local_block_idx % num_blocks_per_row;
            }

            // Wait for inputs to arrive at shared memory
            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            update_phasebit<0>(phasebits, stage);

            // Load input
            rt_bf<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE> A_bf16;
            if constexpr (transpose)
                transpose_load(A_bf16, inputs[stage].A_bf16);
            else
                consumer::load(A_bf16, inputs[stage].A_bf16);
            consumer::sync(1);
            consumer::arrive(inputs_finished[stage]);

            // Quantize
            static_assert(globals::BLOCK_SIZE / 8 == 16);
            static_assert(globals::BLOCK_SIZE / 4 == 32);
            rt_fl<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE> A_fl_full;
            warp::copy(A_fl_full, A_bf16);
            auto &A_fl = *reinterpret_cast<rt_fl<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE / 4>(*)[4]>(&A_fl_full);
            rt_fl<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE / 4> A_fl_abs[4];
            col_vec<rt_fl<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE / 4>> scale[4]; // ortho layout
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                warp::abs(A_fl_abs[i], A_fl[i]);
                warp::row_max(scale[i], A_fl_abs[i]);
                warp::mul(scale[i], scale[i], 0.002232142857f); // 1 / 448
                // Must utilize Blackwell HW for narrowing to ue8m0
                #pragma unroll
                for(int j = 0; j < scale[i].inner_dim; j++) {
                    static_assert(scale[i].inner_dim == 1);
                    fp8e8m0 tmp[2];
                    // Rounding towards pos_inf + finite saturation (https://arxiv.org/pdf/2506.08027)
                    // Simply put, this rounds up to the nearest 2^n
                    asm volatile("{cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;}"
                        : "=h"(reinterpret_cast<fp8e8m0_2 *>(&tmp[0])->__x)
                        : "f"(scale[i][0][j].y), "f"(scale[i][0][j].x)); // careful with the order!
                    scale[i][0][j].x = float(tmp[0]); // After narrowing, convert back for division
                    scale[i][0][j].y = float(tmp[1]);
                }
                warp::div_row(A_fl[i], A_fl[i], scale[i]);
            }
            rt_fp8e4m3<globals::BLOCK_SIZE / 8, globals::BLOCK_SIZE> A_fp8;
            warp::copy(A_fp8, A_fl_full);

            // Store results to shared memory
            consumer::store(outputs.A_fp8, A_fp8);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = (lane_id % 4) * 8 + (lane_id / 4);
                if (idx < 16) {
                    float src = (lane_id % 2 == 0) ? scale[i][0][0].x : scale[i][0][0].y;
                    fp8e8m0 tmp[2];
                    uint32_t st_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&outputs.A_sc.data[0]));
                    asm volatile("{cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;}"
                        : "=h"(reinterpret_cast<fp8e8m0_2 *>(&tmp[0])->__x)
                        : "f"(src), "f"(src));
                    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
                    asm volatile("{st.shared.b8 [%0], %1;}" 
                        :: "r"(st_ptr + warpgroup_id * 256 + idx * 16 + warp_id * 4 + i), 
                           "h"(reinterpret_cast<fp8e8m0_2 *>(&tmp[0])->__x));
                }
            }
            consumer::sync(1);

            // Store results to global memory
            if (consumer::laneid() == 0) {
                tma::store_async(G.A_fp8, outputs.A_fp8, {group_idx, row_block_idx, col_block_idx});
                tma::store_async(G.A_sc, outputs.A_sc, {group_idx, row_block_idx, col_block_idx, 0});
                tma::store_async_read_wait();
            }
            consumer::sync(1);

            // Update phasebit and stage
            stage = (stage + 1) % config::PIPELINE_STAGES;
        }
    }
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<kernel<false>>(m, "kernel",
        &globals::A_bf16,
        &globals::A_fp8,
        &globals::A_sc
    );
    kittens::py::bind_kernel<kernel<true>>(m, "kernel_transpose",
        &globals::A_bf16,
        &globals::A_fp8,
        &globals::A_sc
    );
}
