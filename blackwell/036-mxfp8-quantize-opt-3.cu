/*
    Observation 1:
        DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) 
        DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1)
        DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) 
        DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1) DO NOT USE CLUSTER_DIM(1)

        It was the reason why the kernels suddenly became slow! Like REALLY slow, like 40% slower!
        It changes the way CUDA schedules blocks in a way that reduces the occupancy. Probably because
        it schedules at the granularity of clusters instead of individual blocks. I think there is also
        extra launch setup/teardown overhead for cluster mode.

    Observation 2:
        cp.async.bulk is surprisingly slow compared to cp.async.bulk.tensor, even if we are storing 1D tensor.
        Thus, we should just always use cp.async.bulk.tensor, and never cp.async.bulk

    Observation 3:
        - Let's fix to 204800x2048 matrix, 128x128 tile for this
        - Without calculation logic, we reach 6075.39 GB/s
        - The full thing reaches THE SAME BANDWIDTH. Now we are truly memory bound!!
            - Remember that a little slowdown happens inevitably since calculation logic
              consumes a bit of shared memory bandwidth
    
    Observation 4 (hypothesis, not fully verified):
        - The TK style of "load everything from smem" -> "do it" -> "store everything to smem"
          only works when kernel is compute bound.
        - With memory-bound kernels, this is no longer the case. We should do little bit of 
          shared memory load, calculation, then little bit of shared memory store.
        - Why? Because we don't want to clog up shared memory bandwidth in memory bound workloads.
          We need to ensure gmem <-> smem is always fully saturated, and smem <-> reg should 
          bother this as little as possible.
        ** Note: I changed the load part to full load, and speed wasn't changed. Maybe this isn't the case **
    
    Observation 5
        - Bank conflict is a real thing
        - Changing just the load pattern to non-swizzled degrades perf down to 4030 GB/s
        - Changing load and store pattern to non-swizzled (thus result correct) degrades perf down to 2901 GB/s. 2x slower!
*/

#include <kittens.cuh>
#include <pybind11/pybind11.h>

using namespace kittens;
namespace py = pybind11;

static constexpr int E = 1;
static constexpr int M = 204800;
static constexpr int N = 2048;

// Changing these requires re-writing the kernel
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;
static constexpr int Q_BLOCK_SIZE = 32;

__global__ __launch_bounds__(TILE_M)
void kernel(
    const bf16 *A_bf16,
    const __grid_constant__ CUtensorMap A_bf16_tmap,
    fp8e4m3 *A_fp8,
    const __grid_constant__ CUtensorMap A_fp8_tmap,
    fp8e8m0 *A_sc,
    const __grid_constant__ CUtensorMap A_sc_tmap
) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    uint64_t __shm_base = reinterpret_cast<uint64_t>(&__shm[0]);
    bf16 *A_bf16_smem = reinterpret_cast<bf16*>(((__shm_base + 1023) / 1024) * 1024);
    fp8e4m3 *A_fp8_smem = reinterpret_cast<fp8e4m3*>(A_bf16_smem);
    fp8e8m0 *A_sc_smem = reinterpret_cast<fp8e8m0*>(A_fp8_smem + TILE_M * TILE_N); // naturally fulfills alignment

    // Calculate indices
    int group = blockIdx.z;
    int row = blockIdx.y * TILE_M;
    int col = blockIdx.x * TILE_N;
    int tid = threadIdx.x;

    // Initialize mbarriers
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
    }
    __syncthreads();

    // Initiate the load from global memory
    if (tid == 0) {
        tma::expect_bytes(inputs_arrived, TILE_M * TILE_N * sizeof(bf16));
        asm volatile("{cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, %4}], [%5];}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_bf16_smem))), "l"(&A_bf16_tmap), "r"(col), "r"(row), "r"(group),
               "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived)))
            : "memory");
    }

    // Wait for the load to complete
    asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible
    wait(inputs_arrived, 0);

    // We have 128 threads per block. Each thread handles a row of 128 elements
    constexpr int NUM_Q_BLOCKS = TILE_N / Q_BLOCK_SIZE; // 4
    constexpr int N_PER_Q_BLOCK = TILE_N / 2 / NUM_Q_BLOCKS; // 16
    bf16_2 A_bf16_reg[NUM_Q_BLOCKS][N_PER_Q_BLOCK];
    fp8e8m0 A_sc_reg[NUM_Q_BLOCKS];

    // Load input matrix from shared memory (swizzled)
    #pragma unroll
    for (int i = 0; i < NUM_Q_BLOCKS; i++) {
        int q_block_idx = (i + tid / 8) % NUM_Q_BLOCKS;
        #pragma unroll
        for (int j = 0; j < N_PER_Q_BLOCK; j++) {
            int offset = tid * TILE_N * sizeof(bf16) + // row
                         q_block_idx * Q_BLOCK_SIZE  * sizeof(bf16) + // Q block
                         ((tid + j) % 16) * sizeof(bf16_2); // element within Q block (swizzled)
            asm volatile("{ld.shared.b32 %0, [%1];}"
                : "=r"(*reinterpret_cast<uint32_t *>(&A_bf16_reg[i][j]))
                : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_bf16_smem)) + offset));
        }
    }
    __syncthreads();

    // Perform MXFP8 quantization
    #pragma unroll
    for (int i = 0; i < NUM_Q_BLOCKS; i++) {
        // A group of 8 threads handles the same Q block segment
        int q_block_idx = (i + tid / 8) % NUM_Q_BLOCKS;

        // Calculate absolute maximum
        bf16_2 amax = __habs2(A_bf16_reg[i][0]);
        #pragma unroll
        for (int j = 1; j < N_PER_Q_BLOCK; j++)
            amax = __hmax2(amax, __habs2(A_bf16_reg[i][j]));

        // Compute the scales
        // Must narrow to e8m0, rounding towards positive infinity and saturating to finite, then clamp
        // https://arxiv.org/pdf/2506.08027
        float scale = max(__bfloat162float(__hmax(amax.x, amax.y)) * 0.002232142857f, 0.000000000001f);
        A_sc_reg[q_block_idx].__x = __nv_cvt_float_to_e8m0(scale, __NV_SATFINITE, cudaRoundPosInf); // causes stack frame, but ignorable
        scale = static_cast<float>(A_sc_reg[q_block_idx]); // utilizes the float() operator defined in __nv_fp8x2_e8m0

        // Quantize input matrix and store to share memory
        #pragma unroll
        for (int j = 0; j < N_PER_Q_BLOCK; j++) {
            int offset = tid * TILE_N * sizeof(fp8e4m3) + // row
                         q_block_idx * Q_BLOCK_SIZE * sizeof(fp8e4m3) + // Q block
                         ((tid + j) % 16) * sizeof(fp8e4m3_2); // element within Q block (swizzled)
            fp8e4m3 A_fp8_reg[2] = {
                __nv_fp8_e4m3(__bfloat162float(A_bf16_reg[i][j].x) / scale),
                __nv_fp8_e4m3(__bfloat162float(A_bf16_reg[i][j].y) / scale)
            };
            asm volatile("{st.shared.b16 [%0], %1;}"
                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_fp8_smem)) + offset)
                   "h"(*reinterpret_cast<uint16_t *>(&A_fp8_reg[0])));
        }
    }

    // Store the scales to shared memory. Each thread will access 1 bank, so no need to swizzle,
    // but we do have to follow this complicated layout pattern made by NVIDIA:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
    int scale_offset = (tid % 32) * 16 + // row
                       (tid / 32) * 4; // column
    asm volatile("{st.shared.b32 [%0], %1;}" 
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_sc_smem)) + scale_offset)
           "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[0])));

    // Store to global memory
    asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible
    __syncthreads();
    if (tid == 0) {
        // Since this is the only store, no need to wait for completion
        asm volatile("{cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];}"
            :: "l"(&A_fp8_tmap), "r"(col), "r"(row), "r"(group),
               "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_fp8_smem)))
            : "memory");
        asm volatile("{cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];}"
            :: "l"(&A_sc_tmap), "n"(0), "r"(col / TILE_N), "r"(row / TILE_M), "r"(group),
               "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_sc_smem)))
            : "memory");
        asm volatile("{cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];}"
            :: "l"(&A_sc_tmap), "r"(TILE_N * TILE_M / Q_BLOCK_SIZE / 2), "r"(col / TILE_N), "r"(row / TILE_M), "r"(group),
               "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_sc_smem)) + TILE_M * TILE_N / Q_BLOCK_SIZE / 2)
            : "memory");
    }
}

template <typename T>
__host__ static inline T *get_data_ptr(py::object tensor) {
    // Assumes the following about `tensor`
    // - is a torch.Tensor object
    // - is contiguous
    // - is on device
    // - has the correct shape
    return reinterpret_cast<T *>(tensor.attr("data_ptr")().cast<uintptr_t>());
}

__host__ static inline void launch_kernel(py::object &A_bf16, py::object &A_fp8, py::object &A_sc) {
    CUtensorMap A_bf16_tmap, A_fp8_tmap, A_sc_tmap;

    static constexpr int A_bf16_rank = 3;
    static constexpr int A_fp8_rank = 3;
    static constexpr int A_sc_rank = 4;

    uint64_t A_bf16_shape[A_bf16_rank] = {N, M, E}; // inner-dim first
    uint64_t A_bf16_stride[A_bf16_rank - 1] = {N * sizeof(bf16), M * N * sizeof(bf16)};
    uint32_t A_bf16_smem_shape[A_bf16_rank] = {TILE_N, TILE_M, 1};
    uint32_t A_bf16_smem_stride[A_bf16_rank] = {1, 1, 1};
    
    uint64_t A_fp8_shape[A_fp8_rank] = {N, M, E};
    uint64_t A_fp8_stride[A_fp8_rank - 1] = {N * sizeof(fp8e4m3), M * N * sizeof(fp8e4m3)};
    uint32_t A_fp8_smem_shape[A_fp8_rank] = {TILE_N, TILE_M, 1};
    uint32_t A_fp8_smem_stride[A_fp8_rank] = {1, 1, 1};

    uint64_t A_sc_shape[A_sc_rank] = {TILE_N * TILE_M / Q_BLOCK_SIZE, N / TILE_N, M / TILE_M, E};
    uint64_t A_sc_stride[A_sc_rank - 1] = {TILE_N * TILE_M / Q_BLOCK_SIZE * sizeof(fp8e8m0), N * TILE_M / Q_BLOCK_SIZE * sizeof(fp8e4m3), M * N / Q_BLOCK_SIZE * sizeof(fp8e4m3)};
    uint32_t A_sc_smem_shape[A_sc_rank] = {TILE_N * TILE_M / Q_BLOCK_SIZE / 2, 1, 1, 1}; // divide into 2 TMA stores
    uint32_t A_sc_smem_stride[A_sc_rank] = {1, 1, 1, 1};

    CUCHECK(cuTensorMapEncodeTiled(
        &A_bf16_tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        A_bf16_rank,
        (void *)get_data_ptr<bf16>(A_bf16),
        &A_bf16_shape[0],
        &A_bf16_stride[0],
        &A_bf16_smem_shape[0],
        &A_bf16_smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    CUCHECK(cuTensorMapEncodeTiled(
        &A_fp8_tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        A_fp8_rank,
        (void *)get_data_ptr<fp8e4m3>(A_fp8),
        &A_fp8_shape[0],
        &A_fp8_stride[0],
        &A_fp8_smem_shape[0],
        &A_fp8_smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    CUCHECK(cuTensorMapEncodeTiled(
        &A_sc_tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        A_sc_rank,
        (void *)get_data_ptr<fp8e8m0>(A_sc),
        &A_sc_shape[0],
        &A_sc_stride[0],
        &A_sc_smem_shape[0],
        &A_sc_smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    static constexpr int DYNAMIC_SMEM = TILE_M * TILE_N * sizeof(bf16) + 1024;
    dim3 grid = dim3(N / TILE_N, M / TILE_M, E);
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM));
    kernel<<<grid, TILE_M, DYNAMIC_SMEM>>>(
        get_data_ptr<bf16>(A_bf16), A_bf16_tmap, get_data_ptr<fp8e4m3>(A_fp8), A_fp8_tmap, get_data_ptr<fp8e8m0>(A_sc), A_sc_tmap
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &launch_kernel);
}
