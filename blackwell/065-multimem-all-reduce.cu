/*
    Observations

    - The checks have no effect on speed (obvious)
        - With full check: 304 ~ 318 GB/s
        - With no check: 300 ~ 317 GB/s
    - Using TK abstractions is very slow, must find why:
        - Using R -> S -> G store path: 310.79 GB/s
        - Using R -> G store path: 291 GB/s

    - Using raw asm gets us 3x faster (!), there must be something wrong with TK abstractions
        - Rough implementation gets us 860 GB/s

    - Raw (all using BF16x2, reduce-add):
        - weak vs release vs acquire
            weak: 861.42 GB/s
            relaxed: 845.92 GB/s
            release/acquire: 788.56 GB/s  <---- big slowdown!
        - acc::f32 vs acc:f16
            acc::f32: 864.30 GB/s
            acc::f16: 845.77 GB/s    <-- suprisingly, a little slower (do need to consider the variance)
                                         *note that this is done by removing "acc::f32" qualifier rather than adding acc::f16
        - v1 vs v2 vs v4 (all non-persistent grid, same number of blocks, in-kernel looping for v1 and v2)
            v1: 207.21 GB/s      <-- multimem instruction is blocking & slow; looping multiple multimem instructions is bad; rely on warp scheduling for asynchrony. Also probably due to bad L2 util
            v2: 461.58 GB/s
            v4: 853.81 GB/s
        - v1 vs v2 vs v4 (all non-persistent grid, no looping in-kernel, 2x and 4x number of blocks for v2 and v1 respectively)
            v1: 856.64 GB/s      <-- no difference, no *true* vector-op exists in NVLS it seems. All it matters is asynchrony and overlapping
            v2: 856.71 GB/s
            v4: 853.81 GB/s
        - Persistent grid (looping) vs non-persistent grid (no looping)
            Persistent (148): 796.42 GB/s          <---- typical observation where memory bound op is slightly slower with persistent grid pattern
            Persistent (148 x 2): 846.93 GB/s      <---- technically we can fit in 32 blocks per SM (this barely uses any registers, no SMEM at all), so this definitely helps with occupancy
            Persistent (148 x 3): 851.92 GB/s
            Persistent (148 x 6): 841.16 GB/s
            Non-persistent: 855.62 GB/s
        - multimem.st vs TMA store
            multimem.st: 855.62 GB/s
            TMA store:
        - Tiles ablation
            TK-layout tile : 173.89 GB/s
            Coalesced-layout tile: 840.27 GB/s   <---- explanation on this (long line yay): it striked very weird that TK layout is very slow while naive persistent grid pattern is fast. So my new hypothesis was that NVSwitch combines coalesced access, as that is the only common thing between in-kernel loop version and TK layout which were both very slow: the threads in the warp do not access contiguous memory. So as a test, I do the same things as TK tile except for that a tile is perfectly coalesced (i.e., 128 threads handling contiguous 256 elements at each iteration, and doing 128 iterations to handle 128x256 tile per threadblock). And it turned out to be fast, proving my thoughts
            no tile: 855.62 GB/s

    - Key optimization lessons:
        - Do not use strong memory consistency (relaxed/acquire/acquire) unless memory ordering matters (which does *not* in all reduce)
        - Rely on warp-scheduler-level asynchrony as multimem instructions are blocking & slow
        - Coalesced access is *extremely* important. Discontinuous access leads to serious speed down!
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 128;
    static constexpr int COL_BLOCK_SIZE = 256; // 128 threads take care of a row simultaneously (2 elems per row)

    using tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true, tma::descriptor<tile, 2, false>>;

    parallel_layout tensor;
    const int dev_idx;

    __host__ inline dim3 grid() const { 
        return dim3(tensor.batch() * tensor.depth() * 
                   (tensor.rows() / ROW_BLOCK_SIZE / NUM_DEVICES) *
                   (tensor.cols() / COL_BLOCK_SIZE)); 
        // return dim3(tensor.numel() / NUM_DEVICES / 512);
    }
    __host__ inline int dynamic_shared_memory() const {
        return 0;
        // return static_cast<int>(sizeof(tile) + 1024);
    }
};

__device__ inline int4 get_indices(const globals &G, int task_id) {
    const int num_row_blocks = G.tensor.rows() / globals::NUM_DEVICES / globals::ROW_BLOCK_SIZE;
    const int num_col_blocks = G.tensor.cols() / globals::COL_BLOCK_SIZE;

    int4 indices;

    indices.x = task_id / (G.tensor.depth() * num_row_blocks * num_col_blocks);
    task_id -= indices.x * (G.tensor.depth() * num_row_blocks * num_col_blocks);
    indices.y = task_id / (num_row_blocks * num_col_blocks);
    task_id -= indices.y * (num_row_blocks * num_col_blocks);
    indices.z = task_id / num_col_blocks;
    task_id -= indices.z * num_col_blocks;
    indices.w = task_id;

    if (indices.x >= G.tensor.batch())
        return { -1, -1, -1, -1 };
    else
        return indices;
}

__device__ inline void kernel(const globals &G) {
    // Declare shared memory
    // extern __shared__ int __shm[];
    // uint64_t __shm_base = reinterpret_cast<uint64_t>(&__shm[0]);
    // bf16 *tile = reinterpret_cast<bf16*>(((__shm_base + 1023) / 1024) * 1024);

    // Retrieve indices
    int4 indices = get_indices(G, blockIdx.x);
    const int num_row_blocks = G.tensor.rows() / globals::NUM_DEVICES / globals::ROW_BLOCK_SIZE;
    const int warp_id = warpgroup::warpid();
    const int base_idx =
        indices.x * G.tensor.cols() * G.tensor.rows() * G.tensor.depth() +
        indices.y * G.tensor.cols() * G.tensor.rows() +
        (indices.z + num_row_blocks * G.dev_idx) * globals::ROW_BLOCK_SIZE * G.tensor.cols() + 
        indices.w * globals::COL_BLOCK_SIZE;

    #pragma unroll
    for (int i = 0; i < 128; i++) {
        int idx = base_idx + i * G.tensor.cols() + threadIdx.x * 2;
        bf16_2 tmp;
        multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<bf16_2 *>(&G.tensor.mc_ptr[idx]));
        multimem<bf16_2>::st(reinterpret_cast<bf16_2 *>(&G.tensor.mc_ptr[idx]), tmp);
    }

    // const int N_total = G.tensor.numel();
    // const int N_per_dev = N_total / globals::NUM_DEVICES;
    // constexpr int N_per_inst = 2;
    // constexpr int N_per_iter = config::NUM_THREADS * N_per_inst; // 2048

    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("%d %d %d\n", N_total, N_per_dev, N_per_iter);
    //     printf("%d\n", idx);
    // }

    // const int end = N_per_dev * (G.dev_idx + 1);
        
    // Assume N is divisible by N_per_iter
    // int idx = N_per_dev * G.dev_idx + N_per_iter * blockIdx.x + N_per_inst * threadIdx.x;
    // // for (int idx = N_per_dev * G.dev_idx + N_per_iter * blockIdx.x + N_per_inst * threadIdx.x; idx < end; idx += N_per_iter * gridDim.x) {
    //     bf16_2 tmp;
    //     multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<bf16_2 *>(&G.tensor.mc_ptr[idx]));
    //     multimem<bf16_2>::st(reinterpret_cast<bf16_2 *>(&G.tensor.mc_ptr[idx]), tmp);
    // // }

    // rt_bf<globals::ROW_BLOCK_SIZE / 4, globals::COL_BLOCK_SIZE> intermediate;
    // warpgroup::all_reduce_add(intermediate, G.tensor, 
    //     {indices.x, indices.y, indices.z + G.dev_idx * num_row_blocks, indices.w});

    // // Store back to SMEM
    // warpgroup::store(tile, intermediate);
    // __syncthreads();

    // // Store back to GMEM (no commit required)
    // warpgroup::tma::store_async(G.tensor, tile,
    //     {indices.x, indices.y, indices.z + G.dev_idx * num_row_blocks, indices.w});
}

void entrypoint(kittens::py::TKParallelTensor &tensor) {
    kittens::py::parallel_tensor_check(tensor);

    // TORCH_CHECK(tensor.data_.size(-2) % (globals::ROW_BLOCK_SIZE * globals::NUM_DEVICES) == 0, "tensor.shape[-2] must be divisible by ROW_BLOCK_SIZE * NUM_DEVICES");
    // TORCH_CHECK(tensor.data_.size(-1) % globals::COL_BLOCK_SIZE == 0, "tensor.shape[-1] must be divisible by COL_BLOCK_SIZE");

    globals G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename globals::parallel_layout>(tensor),
        .dev_idx = tensor.local_rank_
    };

    kittens::py::launch_kernel<config, globals, kernel>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_reduce", &entrypoint);
}
