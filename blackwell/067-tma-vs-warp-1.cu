/*
    Curious: would relying on occupancy have similar speed as TMA?

    TMA: 6651.22 GB/s
    Warp asynchrony: 1983.76 GB/s

    Well this was obvious since this experiment does not have to go through the registers at all
*/
#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

// 0: TMA
// 1: Warp
static constexpr int MODE = 1;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS; 
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int BLOCK_SIZE = 128;

    using tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;
    using layout = gl<bf16,  1, 1, -1, -1, tile>;

    layout src;
    layout dst;

    __host__ inline dim3 grid() const {
        if constexpr (MODE == 0) {
            return dim3(src.cols() / BLOCK_SIZE, src.rows() / BLOCK_SIZE);
        } else {
            return dim3(src.cols() * src.rows() / (config::NUM_THREADS * 2));
        }
    }
    __host__ inline int dynamic_shared_memory() const {
        if constexpr (MODE == 0) {
            return sizeof(tile) + 1024;
        } else {
            return 0;
        }
    }
};

__device__ inline void kernel(const globals &G) {
    if constexpr (MODE == 0) {
        // Declare shared memory
        extern __shared__ int __shm[]; 
        tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
        globals::tile &tile = sm_allocator.allocate<globals::tile>();

        const int row = blockIdx.y;
        const int col = blockIdx.x;

        __shared__ semaphore inputs_arrived;
        if (threadIdx.x == 0) {
            init_semaphore(inputs_arrived, 0, 1);
            tma::expect_bytes(inputs_arrived, sizeof(globals::tile));
            tma::load_async(tile, G.src, {row, col}, inputs_arrived);
        }
        __syncthreads();

        wait(inputs_arrived, 0);

        if (threadIdx.x == 0) {
            tma::store_async(G.dst, tile, {row, col});
        }
    } else {
        const int idx = 2 * config::NUM_THREADS * blockIdx.x + 2 * threadIdx.x;
        bf16_2 tmp;
        asm volatile("{ld.weak.global.b32 %0, [%1];}"
            : "=r"(*reinterpret_cast<uint32_t*>(&tmp))
            : "l"(&G.src.raw_ptr[idx])
            : "memory"
        );
        asm volatile("{st.weak.global.b32 [%0], %1;}"
            :
            : "l"(&G.dst.raw_ptr[idx]), "r"(*reinterpret_cast<uint32_t*>(&tmp))
            : "memory"
        );
    }
}

void entrypoint(at::Tensor &src, at::Tensor &dst) {
    globals G {
        .src = kittens::py::tensor_to_gl<globals::layout>(src),
        .dst = kittens::py::tensor_to_gl<globals::layout>(dst)
    };

    kittens::py::launch_kernel<config, globals, kernel>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    m.def("kernel", &entrypoint);
}
