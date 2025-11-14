/*
    BTW,
    - I made a big mistake up to this point; by giving each block max dynamic memory,
      limited the occupancy to 1 block per SM. That is why increasing the tile size
      dramatically increased the bandwidth.
        - On a side note, the value you pass to cudaFuncSetAttribute does not matter.
          It only matters what you pass into <<<>>>, that affects the occupancy

    Motivation: wondering if pipelining helps. That is, is my own pipelining faster
                vs warp scheduler?

    Observations, all using 16384x16384 with 128x128 tile.
        - Baseline (no pipelining): 5048.24 GB/s +- 150 GB/s
        - 4-stage pipeline, no loop: 5100.58 GB/s +- 150 GB/s
            -> this version would stand as direct replacement of warp scheduler logic
            -> this tells that relying on HW warp scheduler vs my own logic does not really matter
        - 4-stage pipeline, loop 2x: similar range as ^
            -> this tests if keeping a block vs having a new block replace is better
            -> seems like not much difference!
        - 4-stage pipeline, loop 4x: 5,000 GB/s +- 150 GB/s
            -> seems slightly slower! maybe better to rely on warp scheduler. but 
               this feels like at the realm of microoptimization
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
static constexpr int PIPELINE_STAGES = 4;
static constexpr int NUM_ITERS = 4;

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

    int row = blockIdx.y * TILE_M * PIPELINE_STAGES * NUM_ITERS;
    int col = blockIdx.x * TILE_N;
    for (int i = 0; i < NUM_ITERS; i++) {
        tma::expect_bytes(inputs_arrived, TILE_M * TILE_N * sizeof(bf16) * PIPELINE_STAGES);
        for (int stage = 0; stage < PIPELINE_STAGES; stage++) {
            asm volatile("{cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3}], [%4];}"
                :: "l"(__cvta_generic_to_shared(t_in_smem + TILE_M * TILE_N * stage)), "l"(&tmap), "r"(col), "r"(row + TILE_M * (i * PIPELINE_STAGES + stage)), "l"(__cvta_generic_to_shared(&inputs_arrived))
                : "memory");
        }
        wait(inputs_arrived, i % 2);
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
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    static constexpr int DYNAMIC_SMEM = TILE_M * TILE_N * 2 * PIPELINE_STAGES + 1024;
    dim3 grid = dim3(N / TILE_N, M / TILE_M / PIPELINE_STAGES / NUM_ITERS);
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM));
    kernel<<<grid, 1, DYNAMIC_SMEM, 0>>>(
        get_data_ptr<bf16>(t_in), tmap, get_data_ptr<bf16>(t_out)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &launch_kernel);
}
