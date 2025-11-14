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

    static constexpr int PIPELINE_STAGES = 6;
};

// Kernel globals
struct globals {
    // Since we are not running any MMAs, we do not need swizzling. Thus use SV
    using A_bf16_st = st<bf16, 128, 128>;
    using A_fp8_st = st<fp8e4m3, 128, 128>;
    using A_sc_sv = sv<fp8e8m0, 512>;

    gl<bf16, 1, -1, -1, -1, A_bf16_st> A_bf16;  // E x M x N
    gl<fp8e4m3, 1, -1, -1, -1, A_fp8_st> A_fp8; // E x M x N
    gl<fp8e8m0, -1, -1, -1, -1, A_sc_sv> A_sc;  // E x (M / 128) x (N / 128) x 512

    __host__ inline dim3 grid() { return dim3(config::SM_COUNT); }
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }

    struct pipeline_inputs {
        A_bf16_st A_bf16;
    };

    struct pipeline_outputs {
        A_fp8_st A_fp8;
        A_sc_sv A_sc;
    };
};

// Kernel implementation
__global__  __launch_bounds__(config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

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
    using consumer = group<config::CONSUMER_WARPGROUPS * WARPGROUP_WARPS>;
    int num_groups = G.A_bf16.depth();
    int num_blocks_per_row = G.A_bf16.cols() / 128;
    int num_blocks_per_col = G.A_bf16.rows() / 128;
    int num_blocks_per_group = num_blocks_per_row * num_blocks_per_col;
    int num_blocks = num_groups * num_blocks_per_group;

    // Declare stage and phasebits for semaphore waits
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup::groupid() == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        // Main loop
        if (warpgroup::laneid() == 0) {
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
        for (int block_idx = blockIdx.x; block_idx < num_blocks; block_idx += gridDim.x) {
            // Compute block indices
            int group_idx = block_idx / num_blocks_per_group;
            int group_local_block_idx = block_idx % num_blocks_per_group;
            int row_block_idx = group_local_block_idx / num_blocks_per_row;
            int col_block_idx = group_local_block_idx % num_blocks_per_row;

            // Wait for inputs to arrive at shared memory
            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            update_phasebit<0>(phasebits, stage);

            // Load input - each thread (256 total) will handle half a row
            bf16_2 A_bf16_2[32];
            uint32_t A_bf16_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&inputs[stage].A_bf16.data[0]));
            // #pragma unroll
            // for (int i = 0; i < 32; i++) {
            //     asm volatile("{ld.shared.b32 %0, [%1];}"
            //         : "=r"(*reinterpret_cast<uint32_t *>(&A_bf16_2[i])) 
            //         : "r"(A_bf16_ptr + (threadIdx.x * 64 + i * 2) * 2));
            // }
            consumer::sync(1);
            consumer::arrive(inputs_finished[stage]);

            // Compute scales
            float A_sc[2];
            #pragma unroll
            for (int scale_idx = 0; scale_idx < 2; scale_idx++) {
                bf16_2 amax = __habs2(A_bf16_2[scale_idx * 16 + 0]);
                #pragma unroll
                for (int i = 1; i < 16; i++) {
                    amax = __hmax2(amax, __habs2(A_bf16_2[scale_idx * 16 + i]));
                }
                A_sc[scale_idx] = __bfloat162float(__hmax(amax.x, amax.y)) * 0.002232142857f; // 1 / 448
            }

            // Narrow the scales to UE8M0 format
            // Must round towards positive infinity and saturate to finite (https://arxiv.org/pdf/2506.08027)
            fp8e8m0_2 A_sc_e8m0_2;
            A_sc_e8m0_2.__x = __nv_cvt_float2_to_e8m0x2(
                *reinterpret_cast<float2 *>(&A_sc[0]), __NV_SATFINITE, cudaRoundPosInf);
            // This utilizes the float2() operator defined in __nv_fp8x2_e8m0
            *reinterpret_cast<float2 *>(&A_sc[0]) = static_cast<float2>(A_sc_e8m0_2);

            // Quantize to FP8E4M3
            fp8e4m3 A_fp8[64];
            #pragma unroll
            for (int scale_idx = 0; scale_idx < 2; scale_idx++) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int idx = scale_idx * 16 + i;
                    A_fp8[idx * 2 + 0] = __nv_fp8_e4m3(__bfloat162float(A_bf16_2[idx].x) / A_sc[scale_idx]);
                    A_fp8[idx * 2 + 1] = __nv_fp8_e4m3(__bfloat162float(A_bf16_2[idx].y) / A_sc[scale_idx]);
                }
            }

            // Store the scales to shared memory. Each thread will access 1 bank, so no need to swizzle,
            // but we do have to follow this complicated layout pattern made by NVIDIA:
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
            uint32_t A_sc_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&outputs.A_sc.data[0]));
            asm volatile("{st.shared.b16 [%1], %0;}" 
                :: "h"(*(uint16_t*)&A_sc_e8m0_2), 
                   "r"(A_sc_ptr + (consumer::warpid() * 32 + warp::laneid()) * 2)
                : "memory");

            // Store the FP8 tile to shared memory
            // uint32_t A_fp8_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&outputs.A_fp8.data[0]));
            // #pragma unroll
            // for (int i = 0; i < 16; i++) {
            //     asm volatile("{st.shared.b32 [%1], %0;}" 
            //         :: "r"(*reinterpret_cast<uint32_t *>(&A_fp8[i * 4])),
            //            "r"(A_fp8_ptr + threadIdx.x * 64 + i * 4)
            //         : "memory");
            // }

            // Store results to global memory
            consumer::sync(1);
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
}
