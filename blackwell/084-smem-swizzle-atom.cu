/*

Notes:
    - The minimum unit of swizzling is swizzle atom. Swizzling happens within that atom only.
    - The HBM is "tiled" over in terms of these swizzle atoms. No reordering happens inter-atom.
    - If inner dim bytes is less than swizzle bytes, TMA will zero pad the swizzle atom
    - BF16 swizzle atom patterns (represented as `atom_idx.index_within_atom`):
        - Swizzle 32B (8x2 atom; 8x16 for BF16):
             0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  1.0  1.1  1.2  1.3  1.4  1.5  1.6  1.7 
             2.0  2.1  2.2  2.3  2.4  2.5  2.6  2.7  3.0  3.1  3.2  3.3  3.4  3.5  3.6  3.7 
             4.0  4.1  4.2  4.3  4.4  4.5  4.6  4.7  5.0  5.1  5.2  5.3  5.4  5.5  5.6  5.7 
             6.0  6.1  6.2  6.3  6.4  6.5  6.6  6.7  7.0  7.1  7.2  7.3  7.4  7.5  7.6  7.7 
             9.0  9.1  9.2  9.3  9.4  9.5  9.6  9.7  8.0  8.1  8.2  8.3  8.4  8.5  8.6  8.7 
            11.0 11.1 11.2 11.3 11.4 11.5 11.6 11.7 10.0 10.1 10.2 10.3 10.4 10.5 10.6 10.7 
            13.0 13.1 13.2 13.3 13.4 13.5 13.6 13.7 12.0 12.1 12.2 12.3 12.4 12.5 12.6 12.7 
            15.0 15.1 15.2 15.3 15.4 15.5 15.6 15.7 14.0 14.1 14.2 14.3 14.4 14.5 14.6 14.7
        - Swizzle 64B (8x4 atom; 8x32 for BF16):
             0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  1.0  1.1  1.2  1.3  1.4  1.5  1.6  1.7  2.0  2.1  2.2  2.3  2.4  2.5  2.6  2.7  3.0  3.1  3.2  3.3  3.4  3.5  3.6  3.7 
             4.0  4.1  4.2  4.3  4.4  4.5  4.6  4.7  5.0  5.1  5.2  5.3  5.4  5.5  5.6  5.7  6.0  6.1  6.2  6.3  6.4  6.5  6.6  6.7  7.0  7.1  7.2  7.3  7.4  7.5  7.6  7.7 
             9.0  9.1  9.2  9.3  9.4  9.5  9.6  9.7  8.0  8.1  8.2  8.3  8.4  8.5  8.6  8.7 11.0 11.1 11.2 11.3 11.4 11.5 11.6 11.7 10.0 10.1 10.2 10.3 10.4 10.5 10.6 10.7 
            13.0 13.1 13.2 13.3 13.4 13.5 13.6 13.7 12.0 12.1 12.2 12.3 12.4 12.5 12.6 12.7 15.0 15.1 15.2 15.3 15.4 15.5 15.6 15.7 14.0 14.1 14.2 14.3 14.4 14.5 14.6 14.7 
            18.0 18.1 18.2 18.3 18.4 18.5 18.6 18.7 19.0 19.1 19.2 19.3 19.4 19.5 19.6 19.7 16.0 16.1 16.2 16.3 16.4 16.5 16.6 16.7 17.0 17.1 17.2 17.3 17.4 17.5 17.6 17.7 
            22.0 22.1 22.2 22.3 22.4 22.5 22.6 22.7 23.0 23.1 23.2 23.3 23.4 23.5 23.6 23.7 20.0 20.1 20.2 20.3 20.4 20.5 20.6 20.7 21.0 21.1 21.2 21.3 21.4 21.5 21.6 21.7 
            27.0 27.1 27.2 27.3 27.4 27.5 27.6 27.7 26.0 26.1 26.2 26.3 26.4 26.5 26.6 26.7 25.0 25.1 25.2 25.3 25.4 25.5 25.6 25.7 24.0 24.1 24.2 24.3 24.4 24.5 24.6 24.7 
            31.0 31.1 31.2 31.3 31.4 31.5 31.6 31.7 30.0 30.1 30.2 30.3 30.4 30.5 30.6 30.7 29.0 29.1 29.2 29.3 29.4 29.5 29.6 29.7 28.0 28.1 28.2 28.3 28.4 28.5 28.6 28.7 
        - Swizzle 128B (8x8 atom; 8x64 for BF16):
             0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  1.0  1.1  1.2  1.3  1.4  1.5  1.6  1.7  2.0  2.1  2.2  2.3  2.4  2.5  2.6  2.7  3.0  3.1  3.2  3.3  3.4  3.5  3.6  3.7  4.0  4.1  4.2  4.3  4.4  4.5  4.6  4.7  5.0  5.1  5.2  5.3  5.4  5.5  5.6  5.7  6.0  6.1  6.2  6.3  6.4  6.5  6.6  6.7  7.0  7.1  7.2  7.3  7.4  7.5  7.6  7.7 
             9.0  9.1  9.2  9.3  9.4  9.5  9.6  9.7  8.0  8.1  8.2  8.3  8.4  8.5  8.6  8.7 11.0 11.1 11.2 11.3 11.4 11.5 11.6 11.7 10.0 10.1 10.2 10.3 10.4 10.5 10.6 10.7 13.0 13.1 13.2 13.3 13.4 13.5 13.6 13.7 12.0 12.1 12.2 12.3 12.4 12.5 12.6 12.7 15.0 15.1 15.2 15.3 15.4 15.5 15.6 15.7 14.0 14.1 14.2 14.3 14.4 14.5 14.6 14.7 
            18.0 18.1 18.2 18.3 18.4 18.5 18.6 18.7 19.0 19.1 19.2 19.3 19.4 19.5 19.6 19.7 16.0 16.1 16.2 16.3 16.4 16.5 16.6 16.7 17.0 17.1 17.2 17.3 17.4 17.5 17.6 17.7 22.0 22.1 22.2 22.3 22.4 22.5 22.6 22.7 23.0 23.1 23.2 23.3 23.4 23.5 23.6 23.7 20.0 20.1 20.2 20.3 20.4 20.5 20.6 20.7 21.0 21.1 21.2 21.3 21.4 21.5 21.6 21.7 
            27.0 27.1 27.2 27.3 27.4 27.5 27.6 27.7 26.0 26.1 26.2 26.3 26.4 26.5 26.6 26.7 25.0 25.1 25.2 25.3 25.4 25.5 25.6 25.7 24.0 24.1 24.2 24.3 24.4 24.5 24.6 24.7 31.0 31.1 31.2 31.3 31.4 31.5 31.6 31.7 30.0 30.1 30.2 30.3 30.4 30.5 30.6 30.7 29.0 29.1 29.2 29.3 29.4 29.5 29.6 29.7 28.0 28.1 28.2 28.3 28.4 28.5 28.6 28.7 
            36.0 36.1 36.2 36.3 36.4 36.5 36.6 36.7 37.0 37.1 37.2 37.3 37.4 37.5 37.6 37.7 38.0 38.1 38.2 38.3 38.4 38.5 38.6 38.7 39.0 39.1 39.2 39.3 39.4 39.5 39.6 39.7 32.0 32.1 32.2 32.3 32.4 32.5 32.6 32.7 33.0 33.1 33.2 33.3 33.4 33.5 33.6 33.7 34.0 34.1 34.2 34.3 34.4 34.5 34.6 34.7 35.0 35.1 35.2 35.3 35.4 35.5 35.6 35.7 
            45.0 45.1 45.2 45.3 45.4 45.5 45.6 45.7 44.0 44.1 44.2 44.3 44.4 44.5 44.6 44.7 47.0 47.1 47.2 47.3 47.4 47.5 47.6 47.7 46.0 46.1 46.2 46.3 46.4 46.5 46.6 46.7 41.0 41.1 41.2 41.3 41.4 41.5 41.6 41.7 40.0 40.1 40.2 40.3 40.4 40.5 40.6 40.7 43.0 43.1 43.2 43.3 43.4 43.5 43.6 43.7 42.0 42.1 42.2 42.3 42.4 42.5 42.6 42.7 
            54.0 54.1 54.2 54.3 54.4 54.5 54.6 54.7 55.0 55.1 55.2 55.3 55.4 55.5 55.6 55.7 52.0 52.1 52.2 52.3 52.4 52.5 52.6 52.7 53.0 53.1 53.2 53.3 53.4 53.5 53.6 53.7 50.0 50.1 50.2 50.3 50.4 50.5 50.6 50.7 51.0 51.1 51.2 51.3 51.4 51.5 51.6 51.7 48.0 48.1 48.2 48.3 48.4 48.5 48.6 48.7 49.0 49.1 49.2 49.3 49.4 49.5 49.6 49.7 
            63.0 63.1 63.2 63.3 63.4 63.5 63.6 63.7 62.0 62.1 62.2 62.3 62.4 62.5 62.6 62.7 61.0 61.1 61.2 61.3 61.4 61.5 61.6 61.7 60.0 60.1 60.2 60.3 60.4 60.5 60.6 60.7 59.0 59.1 59.2 59.3 59.4 59.5 59.6 59.7 58.0 58.1 58.2 58.3 58.4 58.5 58.6 58.7 57.0 57.1 57.2 57.3 57.4 57.5 57.6 57.7 56.0 56.1 56.2 56.3 56.4 56.5 56.6 56.7
*/

#include "kittens.cuh"

using namespace kittens;

using DTYPE = bf16;
static constexpr int M = 8;
static constexpr int N = 64;
static constexpr int TILE_M = M; // single tile
static constexpr int TILE_N = N; // single tile

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
            uint16_t binary_representation = std::bit_cast<uint16_t>(smem[i * TILE_N + j]);
            int idx_within_atom = binary_representation & 0xFF;
            int atom_idx = (binary_representation & 0xFF00) >> 8;
            printf("%2d.%d ", atom_idx, idx_within_atom);
        }
        printf("\n");
    }
}

int main(void) {
    // Allocate host memory
    DTYPE *data_host = new DTYPE[M * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize matrices
    for (int i = 0; i < M * N; ++i) {
        // data_host[i * N + j] = std::bit_cast<DTYPE>(uint16_t(i * N + j));
        // data_host[i * N + j] = std::bit_cast<DTYPE>(uint16_t(i * elem_per_row + (j / elem_per_atom)));
        // data_host[i * N + j] = std::bit_cast<DTYPE>(uint16_t(i * elem_per_row + (j / elem_per_atom)));

        constexpr int elem_per_atom = (128 / 8) / sizeof(DTYPE); // For BF16, 8 elements per atom (128 bits)
        int atom_idx = i / elem_per_atom;
        int idx_within_atom = i % elem_per_atom;

        // Use higher 8 bits for atom_idx, lower 8 bits for idx_within_atom
        uint16_t binary_representation = (atom_idx << 8) | idx_within_atom;
        data_host[i] = std::bit_cast<DTYPE>(binary_representation);
    }
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
        // CU_TENSOR_MAP_SWIZZLE_32B,
        // CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_SWIZZLE_128B,
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
