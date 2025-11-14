/*
    Observations:
        - Only tried out 64x64 with 128B swizzling for convenience
        - No swizzling (from previous tests):
            - 32768x32768 with 64x64 tile: 1165.34 GB/s
            - 16384x16384 with 64x64 tile: 1108.01 GB/s
        - 128B swizzling:
            - 32768x32768 with 64x64 tile: 1163.31 GB/s
            - 16384x16384 with 64x64 tile: 1107.40 GB/s

        ** with fit DM
        - No swizzling (from previous tests):
            - 32768x32768 with 64x64 tile: 6631.39 GB/s
            - 16384x16384 with 64x64 tile: 5109.48 GB/s
        - 128B swizzling:
            - 32768x32768 with 64x64 tile: 6653.17 GB/s
            - 16384x16384 with 64x64 tile: 5149.13 GB/s

        --> As predicted, swizzling itself does NOT increase bandwidth from global to shared,
            which makes sense given that we are loading to every bank anyways in the process.
            Swizzling feature in TMA purely exists to ease user's effort in avoiding bank
            conflicts. If we can avoid bank conflicts without swizzle-loads, then swizzle
            load is unnecessary.
*/

#include <kittens.cuh>
#include <pybind11/pybind11.h>

using namespace kittens;
namespace py = pybind11;

constexpr int rank = 2;
static constexpr int M = 16384
static constexpr int N = 16384
static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;

__global__ void kernel(
    const bf16 *t_in,
    const __grid_constant__ CUtensorMap tmap,
    bf16 *t_out
) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    uint64_t __shm_base = reinterpret_cast<uint64_t>(&__shm[0]);
    bf16 *t_in_smem = reinterpret_cast<bf16*>(((__shm_base + 1023) / 1024) * 1024);

    // Initialize mbarriers
    __shared__ semaphore inputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
    }
    __syncthreads();

    // Initiate the load
    if (threadIdx.x == 0) {
        int row = blockIdx.y * TILE_M;
        int col = blockIdx.x * TILE_N;
        tma::expect_bytes(inputs_arrived, TILE_M * TILE_N * sizeof(bf16));
        asm volatile("{cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3}], [%4];}"
            :: "l"(__cvta_generic_to_shared(t_in_smem)), "l"(&tmap), "r"(col), "r"(row), "l"(__cvta_generic_to_shared(&inputs_arrived))
            : "memory");
    }

    // Wait
    wait(inputs_arrived, 0);

    // Save (for correctness check)
    // if (threadIdx.x == 0) {
    //     int i_base = blockIdx.y * TILE_M;
    //     int j_base = blockIdx.x * TILE_N;
    //     for (int i = 0; i < TILE_M; i++) {
    //         for (int j = 0; j < TILE_N; j++) {
    //             t_out[(i + i_base) * N + (j + j_base)] = t_in_smem[i * TILE_N + j];
    //         }
    //     }
    // }
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

__host__ static inline void launch_kernel(py::object &t_in, py::object &t_out) {
    CUtensorMap tmap;

    uint64_t gmem_shape [2] = {N, M}; // inner-dim first
    uint64_t gmem_stride[1] = {N * sizeof(bf16)};
    uint32_t smem_shape [2] = {TILE_N, TILE_M};
    uint32_t smem_stride[2] = {1, 1};

    CUCHECK(cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        (void *)get_data_ptr<bf16>(t_in),
        &gmem_shape[0],
        &gmem_stride[0],
        &smem_shape[0],
        &smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    static constexpr int DYNAMIC_SMEM = TILE_M * TILE_N * 2 + 1024;
    dim3 grid = dim3(N / TILE_N, M / TILE_M);
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM));
    kernel<<<grid, 1, DYNAMIC_SMEM, 0>>>(
        get_data_ptr<bf16>(t_in), tmap, get_data_ptr<bf16>(t_out)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &launch_kernel);
}
