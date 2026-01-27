/*
    Significant observation: the moment the kernel uses TMEM, occupancy is reduced down to 1.
*/

#include "kittens.cuh"

__global__ void empty_kernel() {}
__global__ void tcgen05_kernel() {
    __shared__ uint32_t tmem_addr;
    asm volatile(
        "{tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [%0], %1;}"
    ::  "l"(reinterpret_cast<uint64_t>(&tmem_addr)), "n"(128)
    );
}

void print_occupancy(const char* name, const void* kernel, int block_size) {
    int max_blocks_per_sm = -1;
    CUDACHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        kernel,
        block_size,
        0
    ));
    printf("%s: Max blocks per SM: %d\n", name, max_blocks_per_sm);
}

int main() {
    const int block_size = 256;
    print_occupancy("empty_kernel", reinterpret_cast<const void*>(empty_kernel), block_size);
    print_occupancy("tcgen05_kernel", reinterpret_cast<const void*>(tcgen05_kernel), block_size);
    return 0;
}
