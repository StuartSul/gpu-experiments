/*
    No swizzling. Just loads and stores

    Observations:
        - It seems like cp.async.bulk.commit_group and cp.async.bulk.wait_group
          are implicitly called at the end of a thread block. Because their presence 
          here does not affect either the speed or correctness, and they are at the
          end of everything here.

        - (full DM --> fit DM)
        - 32768x32768 with 256x256 tile: 6453.76 GB/s --> 6459.02 GB/s
        - 16384x16384 with 256x256 tile: 5658.18 GB/s --> 5636.65 GB/s
        - 32768x32768 with 128x128 tile: 4322.00 GB/s --> 6476.58 GB/s
        - 16384x16384 with 128x128 tile: 3957.28 GB/s --> 5790.32 GB/s
        - 32768x32768 with 64x64 tile: 1868.48 GB/s --> 6259.50 GB/s
        - 16384x16384 with 64x64 tile: 1797.54 GB/s --> 5453.38 GB/s

        - Also tried using CU_TENSOR_MAP_L2_PROMOTION_L2_256B.
          Surprisingly, it decreases the bandwidth by about 2~3%
        - Just in case I also tried CU_TENSOR_MAP_L2_PROMOTION_L2_64B.
          This neither increase nor decrease the bandwidth. I think
          it's safe to stick with no L2 promotion
*/
#include <kittens.cuh>
#include <pybind11/pybind11.h>

using namespace kittens;
namespace py = pybind11;

constexpr int rank = 2;
static constexpr int M = 16384;
static constexpr int N = 16384;
static constexpr int TILE_M = 128;
static constexpr int TILE_N = 128;

__global__ void kernel(
    const bf16 *t_in,
    const __grid_constant__ CUtensorMap tmap_in,
    bf16 *t_out,
    const __grid_constant__ CUtensorMap tmap_out
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

    // Calculate block index
    int row = blockIdx.y * TILE_M;
    int col = blockIdx.x * TILE_N;

    // Initiate the load
    if (threadIdx.x == 0) {
        tma::expect_bytes(inputs_arrived, TILE_M * TILE_N * sizeof(bf16));
        asm volatile("{cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3}], [%4];}"
            :: "l"(__cvta_generic_to_shared(t_in_smem)), "l"(&tmap_in), "r"(col), "r"(row), "l"(__cvta_generic_to_shared(&inputs_arrived))
            : "memory");
    }

    // Wait
    wait(inputs_arrived, 0);

    // Store
    if (threadIdx.x == 0) {
        asm volatile("{cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];}"
            :: "l"(&tmap_out), "r"(col), "r"(row), "l"(__cvta_generic_to_shared(t_in_smem))
            : "memory");
        asm volatile("{cp.async.bulk.commit_group;}");
        asm volatile("{cp.async.bulk.wait_group %0;}"
            :: "n"(0)
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

__host__ static inline void launch_kernel(py::object &t_in, py::object &t_out) {
    CUtensorMap tmap_in, tmap_out;

    uint64_t gmem_shape [2] = {N, M}; // inner-dim first
    uint64_t gmem_stride[1] = {N * sizeof(bf16)};
    uint32_t smem_shape [2] = {TILE_N, TILE_M};
    uint32_t smem_stride[2] = {1, 1};

    CUCHECK(cuTensorMapEncodeTiled(
        &tmap_in,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        (void *)get_data_ptr<bf16>(t_in),
        &gmem_shape[0],
        &gmem_stride[0],
        &smem_shape[0],
        &smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));
    CUCHECK(cuTensorMapEncodeTiled(
        &tmap_out,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        (void *)get_data_ptr<bf16>(t_out),
        &gmem_shape[0],
        &gmem_stride[0],
        &smem_shape[0],
        &smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    static constexpr int DYNAMIC_SMEM = TILE_M * TILE_N * 2 + 1024;
    dim3 grid = dim3(N / TILE_N, M / TILE_M);
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM));
    kernel<<<grid, 1, DYNAMIC_SMEM, 0>>>(
        get_data_ptr<bf16>(t_in), tmap_in, get_data_ptr<bf16>(t_out), tmap_out
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &launch_kernel);
}
