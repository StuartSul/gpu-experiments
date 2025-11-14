/*
    Minimal TMA store example, no swizzling
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
    bf16 *t_out,
    const __grid_constant__ CUtensorMap tmap
) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    uint64_t __shm_base = reinterpret_cast<uint64_t>(&__shm[0]);
    bf16 *t_out_smem = reinterpret_cast<bf16*>(((__shm_base + 1023) / 1024) * 1024);

    // Load
    if (threadIdx.x == 0) {
        for (int i = 0; i < TILE_M; i++) {
            for (int j = 0; j < TILE_N; j += 2) {
                uint32_t tmp = *reinterpret_cast<const uint32_t*>(&t_in[i * TILE_N + j]);
                asm volatile("{st.shared.b32 [%1], %0;}"
                    :: "r"(tmp), "l"(__cvta_generic_to_shared(&t_out_smem[i * TILE_N + j])));
            }
        }
    }

    // Store
    if (threadIdx.x == 0) {
        int row = 0 * TILE_M;
        int col = 0 * TILE_N;
        asm volatile("{cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];}"
            :: "l"(&tmap), "r"(col), "r"(row), "l"(__cvta_generic_to_shared(t_out_smem))
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
    CUtensorMap tmap;

    uint64_t gmem_shape [2] = {N, M}; // inner-dim first
    uint64_t gmem_stride[1] = {N * sizeof(bf16)};
    uint32_t smem_shape [2] = {TILE_N, TILE_M};
    uint32_t smem_stride[2] = {1, 1};

    CUCHECK(cuTensorMapEncodeTiled(
        &tmap,
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

    static constexpr int DYNAMIC_SMEM = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SMEM));
    kernel<<<1, 1, DYNAMIC_SMEM, 0>>>(
        get_data_ptr<bf16>(t_in), get_data_ptr<bf16>(t_out), tmap
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &launch_kernel);
}
