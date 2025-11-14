/*
    Doing both normal and transposed quantize at once!

    For 204800 x 2048 input matrix,
    since normal quantization runs at about 209 us, and doing both has
    4.0625 / 3.03125 = 34% more memory movement, the optimal is at about 280 us.

    Observation:
        - At least for this kernel, cp.async.bulk.wait_group.read seems to
          have almost no benefit over cp.async.bulk.wait_group.

    Benchmarks:
        - Giving full, independent shared memory for all tiles: ** 375 us **
        - Giving indendent shared memory for BF16 input, and overlap normal & transposed outputs: ** 320 us **
            - Requires bulk wait between normal & transposed, with less shared memory -> more occupancy
        - Everything overlaps in BF16 tile: 485 us
            - Requires 2x registers, with less shared memory --> might increase or decrease occupancy

    Conclusion:
        - Finding the middle ground is impt
        - Occupancy is everything

    Also, verified that with either EMIT_T OR EMIT_NT off, it returns to original speed.
*/

#include <kittens.cuh>
#include <pybind11/pybind11.h>

using namespace kittens;
namespace py = pybind11;

static constexpr int E = 1;
static constexpr int M = 204800;
static constexpr int N = 2048;

static constexpr int EMIT_NT = true;
static constexpr bool EMIT_T = true;

// Changing these requires re-writing the kernel
static constexpr int TILE_SIZE = 128;
static constexpr int Q_BLOCK_SIZE = 32;

__global__ __launch_bounds__(TILE_SIZE)
void kernel(
    const bf16 *A_bf16,
    const __grid_constant__ CUtensorMap A_bf16_tmap,
    fp8e4m3 * const A_fp8,
    const __grid_constant__ CUtensorMap A_fp8_tmap,
    fp8e8m0 * const A_sc,
    const __grid_constant__ CUtensorMap A_sc_tmap,
    fp8e4m3 * const A_t_fp8,
    const __grid_constant__ CUtensorMap A_t_fp8_tmap,
    fp8e8m0 * const A_t_sc,
    const __grid_constant__ CUtensorMap A_t_sc_tmap
) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    uint64_t __shm_base = reinterpret_cast<uint64_t>(&__shm[0]);
    bf16 *A_bf16_smem = reinterpret_cast<bf16*>(((__shm_base + 1023) / 1024) * 1024); // if this is aligned, everything else is aligned
    fp8e4m3 *A_fp8_smem, *A_t_fp8_smem;
    fp8e8m0 *A_sc_smem, *A_t_sc_smem;
    if constexpr (EMIT_NT && EMIT_T) {
        A_fp8_smem = reinterpret_cast<fp8e4m3*>(A_bf16_smem + TILE_SIZE * TILE_SIZE);
        A_sc_smem = reinterpret_cast<fp8e8m0*>(A_fp8_smem + TILE_SIZE * TILE_SIZE);
        A_t_fp8_smem = reinterpret_cast<fp8e4m3*>(A_fp8_smem);
        A_t_sc_smem = reinterpret_cast<fp8e8m0*>(A_sc_smem);
    } else if constexpr (EMIT_NT) {
        A_fp8_smem = reinterpret_cast<fp8e4m3*>(A_bf16_smem);
        A_sc_smem = reinterpret_cast<fp8e8m0*>(A_fp8_smem + TILE_SIZE * TILE_SIZE);
    } else if constexpr (EMIT_T) {
        A_t_fp8_smem = reinterpret_cast<fp8e4m3*>(A_bf16_smem);
        A_t_sc_smem = reinterpret_cast<fp8e8m0*>(A_t_fp8_smem + TILE_SIZE * TILE_SIZE);
    }

    // Calculate indices
    int group = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE;
    int col = blockIdx.x * TILE_SIZE;
    int tid = threadIdx.x;

    // Initialize mbarrier and initiate TMA load
    __shared__ semaphore inputs_arrived;
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, TILE_SIZE * TILE_SIZE * sizeof(bf16));
        asm volatile("{cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, %4}], [%5];}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_bf16_smem))), "l"(&A_bf16_tmap), "r"(col), "r"(row), "r"(group),
               "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived)))
            : "memory");
    }
    __syncthreads();

    // Wait for the TMA load to complete
    asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible
    wait(inputs_arrived, 0);

    // We have 128 threads per block. Each thread handles a row of 128 elements
    constexpr int NUM_Q_BLOCKS = TILE_SIZE / Q_BLOCK_SIZE; // 4
    constexpr int N_PER_Q_BLOCK = TILE_SIZE / 2 / NUM_Q_BLOCKS; // 16

    // Handle non-transposed case
    if constexpr (EMIT_NT) {
        bf16_2 A_bf16_reg[NUM_Q_BLOCKS][N_PER_Q_BLOCK];
        fp8e8m0 A_sc_reg[NUM_Q_BLOCKS];

        // Destination tile row this thread will handle
        int tile_row = tid;

        // Load input matrix from shared memory (swizzled)
        #pragma unroll
        for (int i = 0; i < NUM_Q_BLOCKS; i++) {
            int q_block_idx = (i + tid / 8) % NUM_Q_BLOCKS;
            #pragma unroll
            for (int j = 0; j < N_PER_Q_BLOCK; j++) {
                int tile_col = q_block_idx * Q_BLOCK_SIZE + ((tid + j) * 2) % Q_BLOCK_SIZE;
                int offset = (tile_row * TILE_SIZE + tile_col) * sizeof(bf16);
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
                int tile_col = q_block_idx * Q_BLOCK_SIZE + ((tid + j) * 2) % Q_BLOCK_SIZE;
                int offset = (tile_row * TILE_SIZE + tile_col) * sizeof(fp8e4m3);
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
        int scale_offset = (tile_row % 32) * 16 + // row
                           (tile_row / 32) * 4; // column
        asm volatile("{st.shared.b32 [%0], %1;}" 
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_sc_smem)) + scale_offset)
            "r"(*reinterpret_cast<uint32_t *>(&A_sc_reg[0])));

        // Store to global memory
        asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible
        __syncthreads();
        if (tid == 0) {
            asm volatile("{cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];}"
                :: "l"(&A_fp8_tmap), "r"(col), "r"(row), "r"(group),
                "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_fp8_smem)))
                : "memory");
            asm volatile("{cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];}"
                :: "l"(&A_sc_tmap), "n"(0), "r"(col / TILE_SIZE), "r"(row / TILE_SIZE), "r"(group),
                "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_sc_smem)))
                : "memory");
            asm volatile("{cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];}"
                :: "l"(&A_sc_tmap), "r"(TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE / 2), "r"(col / TILE_SIZE), "r"(row / TILE_SIZE), "r"(group),
                "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_sc_smem)) + TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE / 2)
                : "memory");

            // If we also handle transposed case, must wait for shared memory to be free
            if constexpr (EMIT_T) {
                asm volatile("{cp.async.bulk.commit_group;}");
                asm volatile("{cp.async.bulk.wait_group.read %0;}" :: "n"(0) : "memory");
            }
        }
    }

    // Handle transposed case
    if constexpr (EMIT_T) {
        bf16_2 A_t_bf16_reg[NUM_Q_BLOCKS][N_PER_Q_BLOCK];
        fp8e8m0 A_t_sc_reg[NUM_Q_BLOCKS];

        // Destination tile row this thread will handle
        int tile_row = (tid % 64) * 2 + tid / 64;

        // Load input matrix from shared memory (swizzled)
        #pragma unroll
        for (int i = 0; i < NUM_Q_BLOCKS; i++) {
            int q_block_idx = (i + tid / 8) % NUM_Q_BLOCKS;
            #pragma unroll
            for (int j = 0; j < N_PER_Q_BLOCK / 2; j++) {
                int tile_col = q_block_idx * Q_BLOCK_SIZE + ((tid + j) * 4) % Q_BLOCK_SIZE;
                int offset = (tile_row + tile_col * TILE_SIZE) * sizeof(bf16);
                #pragma unroll
                for (int k = 0; k < 2; k++) {
                    asm volatile("{ld.shared.b16 %0, [%1];}" // can't do b32 packed load
                        : "=h"(*reinterpret_cast<uint16_t*>(&A_t_bf16_reg[i][j * 2 + k].x))
                        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_bf16_smem)) + offset));
                    offset += TILE_SIZE * sizeof(bf16);
                    asm volatile("{ld.shared.b16 %0, [%1];}"
                        : "=h"(*reinterpret_cast<uint16_t*>(&A_t_bf16_reg[i][j * 2 + k].y))
                        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_bf16_smem)) + offset));
                    offset += TILE_SIZE * sizeof(bf16);
                }
            }
        }
        __syncthreads();
    
        // Perform MXFP8 quantization
        #pragma unroll
        for (int i = 0; i < NUM_Q_BLOCKS; i++) {
            // A group of 8 threads handles the same Q block segment
            int q_block_idx = (i + tid / 8) % NUM_Q_BLOCKS;
    
            // Calculate absolute maximum
            bf16_2 amax = __habs2(A_t_bf16_reg[i][0]);
            #pragma unroll
            for (int j = 1; j < N_PER_Q_BLOCK; j++)
                amax = __hmax2(amax, __habs2(A_t_bf16_reg[i][j]));
    
            // Compute the scales
            // Must narrow to e8m0, rounding towards positive infinity and saturating to finite, then clamp
            // https://arxiv.org/pdf/2506.08027
            float scale = max(__bfloat162float(__hmax(amax.x, amax.y)) * 0.002232142857f, 0.000000000001f);
            A_t_sc_reg[q_block_idx].__x = __nv_cvt_float_to_e8m0(scale, __NV_SATFINITE, cudaRoundPosInf); // causes stack frame, but ignorable
            scale = static_cast<float>(A_t_sc_reg[q_block_idx]); // utilizes the float() operator defined in __nv_fp8x2_e8m0
    
            // Quantize input matrix and store to share memory
            #pragma unroll
            for (int j = 0; j < N_PER_Q_BLOCK / 2; j++) {
                int tile_col = q_block_idx * Q_BLOCK_SIZE + ((tid + j) * 4) % Q_BLOCK_SIZE;
                int offset = (tile_row * TILE_SIZE + tile_col) * sizeof(fp8e4m3);
                fp8e4m3 A_fp8_reg[4] = {
                    __nv_fp8_e4m3(__bfloat162float(A_t_bf16_reg[i][j * 2 + 0].x) / scale),
                    __nv_fp8_e4m3(__bfloat162float(A_t_bf16_reg[i][j * 2 + 0].y) / scale),
                    __nv_fp8_e4m3(__bfloat162float(A_t_bf16_reg[i][j * 2 + 1].x) / scale),
                    __nv_fp8_e4m3(__bfloat162float(A_t_bf16_reg[i][j * 2 + 1].y) / scale)
                };
                asm volatile("{st.shared.b32 [%0], %1;}"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_t_fp8_smem)) + offset)
                       "r"(*reinterpret_cast<uint32_t *>(&A_fp8_reg[0])));
            }
        }
    
        // Store the scales to shared memory. Each thread will access 1 bank, so no need to swizzle,
        // but we do have to follow this complicated layout pattern made by NVIDIA:
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
        int scale_offset = (tile_row % 32) * 16 + // row
                           (tile_row / 32) * 4; // column
        asm volatile("{st.shared.b32 [%0], %1;}" 
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_t_sc_smem)) + scale_offset)
               "r"(*reinterpret_cast<uint32_t *>(&A_t_sc_reg[0])));
    
        // Store to global memory
        asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible
        __syncthreads();
        if (tid == 0) {
            // Since this is the only store, no need to wait for completion
            asm volatile("{cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];}"
                :: "l"(&A_t_fp8_tmap), "r"(row), "r"(col), "r"(group),
                   "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_t_fp8_smem)))
                : "memory");
            asm volatile("{cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];}"
                :: "l"(&A_t_sc_tmap), "n"(0), "r"(row / TILE_SIZE), "r"(col / TILE_SIZE), "r"(group),
                   "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_t_sc_smem)))
                : "memory");
            asm volatile("{cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];}"
                :: "l"(&A_t_sc_tmap), "r"(TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE / 2), "r"(row / TILE_SIZE), "r"(col / TILE_SIZE), "r"(group),
                   "r"(static_cast<uint32_t>(__cvta_generic_to_shared(A_t_sc_smem)) + TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE / 2)
                : "memory");
        }
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

__host__ static inline void launch_kernel(
    py::object &A_bf16,
    py::object &A_fp8,
    py::object &A_sc,
    py::object &A_t_fp8,
    py::object &A_t_sc
) {
    CUtensorMap A_bf16_tmap, A_fp8_tmap, A_sc_tmap, A_t_fp8_tmap, A_t_sc_tmap;

    static constexpr int input_rank = 3;
    static constexpr int scale_rank = 4;

    uint64_t input_global_shape[input_rank] = {N, M, E}; // inner-dim first
    uint64_t input_transposed_global_shape[input_rank] = {M, N, E};
    uint64_t scale_global_shape[scale_rank] = {TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE, N / TILE_SIZE, M / TILE_SIZE, E};
    uint64_t scale_transposed_global_shape[scale_rank] = {TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE, M / TILE_SIZE, N / TILE_SIZE, E};
    uint32_t input_smem_shape[input_rank] = {TILE_SIZE, TILE_SIZE, 1};
    uint32_t input_smem_stride[input_rank] = {1, 1, 1};
    uint32_t scale_smem_shape[scale_rank] = {TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE / 2, 1, 1, 1}; // divide into 2 TMA stores
    uint32_t scale_smem_stride[scale_rank] = {1, 1, 1, 1};

    uint64_t A_bf16_stride[input_rank - 1] = {N * sizeof(bf16), M * N * sizeof(bf16)};
    uint64_t A_fp8_stride[input_rank - 1] = {N * sizeof(fp8e4m3), M * N * sizeof(fp8e4m3)};    
    uint64_t A_sc_stride[scale_rank - 1] = {TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE * sizeof(fp8e8m0), N * TILE_SIZE / Q_BLOCK_SIZE * sizeof(fp8e8m0), M * N / Q_BLOCK_SIZE * sizeof(fp8e8m0)};
    uint64_t A_t_fp8_stride[input_rank - 1] = {M * sizeof(fp8e4m3), N * M * sizeof(fp8e4m3)};
    uint64_t A_t_sc_stride[scale_rank - 1] = {TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE * sizeof(fp8e8m0), M * TILE_SIZE / Q_BLOCK_SIZE * sizeof(fp8e8m0), N * M / Q_BLOCK_SIZE * sizeof(fp8e8m0)};

    CUCHECK(cuTensorMapEncodeTiled(
        &A_bf16_tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        input_rank, (void *)get_data_ptr<bf16>(A_bf16),
        &input_global_shape[0], &A_bf16_stride[0],
        &input_smem_shape[0], &input_smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    if constexpr (EMIT_NT) {
        CUCHECK(cuTensorMapEncodeTiled(
            &A_fp8_tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            input_rank, (void *)get_data_ptr<fp8e4m3>(A_fp8),
            &input_global_shape[0], &A_fp8_stride[0],
            &input_smem_shape[0], &input_smem_stride[0],
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        ));
        CUCHECK(cuTensorMapEncodeTiled(
            &A_sc_tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            scale_rank, (void *)get_data_ptr<fp8e8m0>(A_sc),
            &scale_global_shape[0], &A_sc_stride[0],
            &scale_smem_shape[0], &scale_smem_stride[0],
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        ));
    }
    if constexpr (EMIT_T) {
        CUCHECK(cuTensorMapEncodeTiled(
            &A_t_fp8_tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            input_rank, (void *)get_data_ptr<fp8e4m3>(A_t_fp8),
            &input_transposed_global_shape[0], &A_t_fp8_stride[0],
            &input_smem_shape[0], &input_smem_stride[0],
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        ));
        CUCHECK(cuTensorMapEncodeTiled(
            &A_t_sc_tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            scale_rank, (void *)get_data_ptr<fp8e8m0>(A_t_sc),
            &scale_transposed_global_shape[0], &A_t_sc_stride[0],
            &scale_smem_shape[0], &scale_smem_stride[0],
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        ));
    }

    dim3 grid = dim3(N / TILE_SIZE, M / TILE_SIZE, E);
    static constexpr int DYNAMIC_SMEM_NTAT = TILE_SIZE * TILE_SIZE * sizeof(bf16) + 1024 + // input matrix
                                             TILE_SIZE * TILE_SIZE * sizeof(fp8e4m3) + // output matrix (no alignment required)
                                             TILE_SIZE * TILE_SIZE / Q_BLOCK_SIZE * sizeof(fp8e8m0); // scale matrix (no alignment required)
    static constexpr int DYNAMIC_SMEM_NTOT = TILE_SIZE * TILE_SIZE * sizeof(bf16) + 1024; // input matrix
    if constexpr (EMIT_NT && EMIT_T) {
        CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM_NTAT));
        kernel<<<grid, TILE_SIZE, DYNAMIC_SMEM_NTAT>>>(
            get_data_ptr<bf16>(A_bf16), A_bf16_tmap, 
            get_data_ptr<fp8e4m3>(A_fp8), A_fp8_tmap, 
            get_data_ptr<fp8e8m0>(A_sc), A_sc_tmap,
            get_data_ptr<fp8e4m3>(A_t_fp8), A_t_fp8_tmap,
            get_data_ptr<fp8e8m0>(A_t_sc), A_t_sc_tmap
        );
    } else {
        CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM_NTOT));
        kernel<<<grid, TILE_SIZE, DYNAMIC_SMEM_NTOT>>>(
            get_data_ptr<bf16>(A_bf16), A_bf16_tmap, 
            get_data_ptr<fp8e4m3>(A_fp8), A_fp8_tmap, 
            get_data_ptr<fp8e8m0>(A_sc), A_sc_tmap,
            get_data_ptr<fp8e4m3>(A_t_fp8), A_t_fp8_tmap,
            get_data_ptr<fp8e8m0>(A_t_sc), A_t_sc_tmap
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &launch_kernel);
}
