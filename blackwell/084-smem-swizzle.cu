/*
    Observations:

    - If inner dim bytes is less than swizzle bytes, TMA will zero pad the swizzle atom
*/

#include "kittens.cuh"

using namespace kittens;

using DTYPE = bf16;
static constexpr int M = 8;
static constexpr int N = 8;
static constexpr int TILE_M = 8;
static constexpr int TILE_N = 8;

__global__ void kernel(const __grid_constant__ CUtensorMap tmap) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    uint64_t __shm_base = reinterpret_cast<uint64_t>(&__shm[0]);
    DTYPE *smem = reinterpret_cast<DTYPE*>(((__shm_base + 1023) / 1024) * 1024);

    // Initialize mbarriers
    __shared__ semaphore inputs_arrived;
    init_semaphore(inputs_arrived, 0, 1);

    // Load
    const int row = TILE_M * 0;
    const int col = TILE_N * 0;
    tma::expect_bytes(inputs_arrived, TILE_M * TILE_N * sizeof(DTYPE));
    asm volatile("{cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3}], [%4];}"
        :: "l"(__cvta_generic_to_shared(smem)), "l"(&tmap), "r"(row), "r"(col), "l"(__cvta_generic_to_shared(&inputs_arrived))
        : "memory");
    wait(inputs_arrived, 0);

    // Inspect
    #pragma unroll
    for (int i = 0; i < TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < TILE_N; j++) {
            printf("%5d ", std::bit_cast<uint16_t>(smem[i * TILE_N + j]));
        }
        printf("\n");
    }
}

int main(void) {
    // Allocate host memory
    DTYPE *data_host = new DTYPE[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) 
        data_host[i] = std::bit_cast<DTYPE>(uint16_t(i));
    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    DTYPE *data_device;
    CUDACHECK(cudaMalloc(&data_device, M * N * sizeof(DTYPE)));
    std::cout << "Allocated device memory" << std::endl;

    // Copy to device memory
    CUDACHECK(cudaMemcpy(data_device, data_host, M * N * sizeof(DTYPE), cudaMemcpyHostToDevice));
    std::cout << "Copied matrices to device" << std::endl;

    // Generate tensor descriptor
    CUtensorMap tmap;
    static constexpr int rank = 2;
    uint64_t gmem_shape [2] = {N, M}; // inner-dim first!
    uint64_t gmem_stride[1] = {N * sizeof(bf16)};
    uint32_t smem_shape [2] = {TILE_N, TILE_M};
    uint32_t smem_stride[2] = {1, 1};
    CUCHECK(cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        (void *)data_device,
        &gmem_shape[0],
        &gmem_stride[0],
        &smem_shape[0],
        &smem_stride[0],
        CU_TENSOR_MAP_INTERLEAVE_NONE,

        // --------------------------------------
        // SWIZZLE_32B requires the SMEM inner dimension to be <= 32 bytes
        // SWIZZLE_64B requires the SMEM inner dimension to be <= 64 bytes
        // SWIZZLE_128B* require the SMEM inner dimension to be <= 128 bytes
        // --------------------------------------
        // CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_SWIZZLE_32B,
        // CU_TENSOR_MAP_SWIZZLE_64B,
        // CU_TENSOR_MAP_SWIZZLE_128B,
        // --------------------------------------

        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    ));

    // Set dynamic SMEM
    constexpr size_t DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY));

    // Launch kernel
    kernel<<<1, 1, DYNAMIC_SHARED_MEMORY, 0>>>(tmap);
    CUDACHECK(cudaDeviceSynchronize());

    return 0;
}
