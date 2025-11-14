/*
    Goal: implement fast single-node all2all collective communication

    Benchmarks (N: 131072, H: 128, D: 128, 8 GPUs, rank 0 only, Unidirectional NVL bandwidth):

        NCCL: 319.01 GB/s
        Naive 1-stage persistent-grid: 642.21 GB/s
        Naive 2-stage persistent-grid: 630.59 GB/s
        Naive 4-stage persistent-grid: 628.76 GB/s
        Naive 8-stage persistent-grid: 632.25 GB/s
        Naive 16-stage persistent-grid: 627.79 GB/s
        Stage-parllel 1-stage persistent-grid: 645.83 GB/s
        Stage-parllel 2-stage persistent-grid: 615.85 GB/s
        Stage-parllel 4-stage persistent-grid: 641.20 GB/s
        Stage-parllel 8-stage persistent-grid: 627.09 GB/s
        Stage-parllel 16-stage persistent-grid: 557.56 GB/s
        Stage-parllel 32-stage persistent-grid: 496.32 GB/s
        Non-persistent-grid: 693.37 GB/s

    Conclusion:
        - It always seems to be the case that for memory-bound workloads, non-persistent-grid is faster.
        - 16x128 vs 128x128 only gives slight degradation (about 10 GB/s). As long as you have enough SM occupancy, you are good.
*/

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int ROW_BLOCK_SIZE = 16;
    static constexpr int COL_BLOCK_SIZE = 128;

    using tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1, tma::descriptor<tile, 2, false>>, NUM_DEVICES, false>;

    parallel_layout src;
    parallel_layout dst;
    const int dev_idx;

    __host__ inline dim3 grid() const { 
        return dim3(src.batch() *
                    src.depth() * 
                    (src.rows() / globals::ROW_BLOCK_SIZE) *
                    (src.cols() / globals::COL_BLOCK_SIZE)); 
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(tile) + 1024);
    }
};

__device__ inline int4 get_indices(const globals &G, int task_id) {
    int4 indices;

    indices.x = task_id / (G.src.depth() * (G.src.rows() / globals::ROW_BLOCK_SIZE) * (G.src.cols() / globals::COL_BLOCK_SIZE));
    task_id -= indices.x * (G.src.depth() * (G.src.rows() / globals::ROW_BLOCK_SIZE) * (G.src.cols() / globals::COL_BLOCK_SIZE));
    indices.y = task_id / ((G.src.rows() / globals::ROW_BLOCK_SIZE) * (G.src.cols() / globals::COL_BLOCK_SIZE));
    task_id -= indices.y * ((G.src.rows() / globals::ROW_BLOCK_SIZE) * (G.src.cols() / globals::COL_BLOCK_SIZE));
    indices.z = task_id / (G.src.cols() / globals::COL_BLOCK_SIZE);
    task_id -= indices.z * (G.src.cols() / globals::COL_BLOCK_SIZE);
    indices.w = task_id;

    if (indices.x >= G.src.batch())
        return { -1, -1, -1, -1 };
    else
        return indices;
}

template <int SCATTER_AXIS = 2, int GATHER_AXIS = 1>
__device__ inline void kernel(const globals &G) {
    static_assert(SCATTER_AXIS < 4 && GATHER_AXIS < 4, "Scatter and gather axes must be between 0 and 3");
    static_assert(SCATTER_AXIS != GATHER_AXIS, "Scatter and gather axes must be different");

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::tile &tile = allocator.allocate<globals::tile>();

    // Retrieve indices
    int4 indices = get_indices(G, blockIdx.x);

    // Set up mbarrier
    __shared__ semaphore inputs_arrived;
    init_semaphore(inputs_arrived, 0, 1);

    // Initiate the load
    tma::expect_bytes(inputs_arrived, sizeof(globals::tile));
    asm volatile("{cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, %4, %5}], [%6];}"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&tile[0]))), 
            "l"(G.src[G.dev_idx].template get_tma<globals::tile, 2>()), 
            "r"(indices.w * globals::COL_BLOCK_SIZE), "r"(indices.z * globals::ROW_BLOCK_SIZE), "r"(indices.y), "r"(indices.x),
            "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived)))
        : "memory");

    // Calculate the destination indices
    int dst_dev_idx;
    if constexpr (SCATTER_AXIS == 0) {
        int scatter_blocks_per_dev = G.src.batch() / globals::NUM_DEVICES;
        dst_dev_idx = indices.x / scatter_blocks_per_dev;
        indices.x -= scatter_blocks_per_dev * dst_dev_idx;
    } else if constexpr (SCATTER_AXIS == 1) {
        int scatter_blocks_per_dev = G.src.depth() / globals::NUM_DEVICES;
        dst_dev_idx = indices.y / scatter_blocks_per_dev;
        indices.y -= scatter_blocks_per_dev * dst_dev_idx;
    } else if constexpr (SCATTER_AXIS == 2) {
        int scatter_blocks_per_dev = G.src.rows() / globals::ROW_BLOCK_SIZE / globals::NUM_DEVICES;
        dst_dev_idx = indices.z / scatter_blocks_per_dev;
        indices.z -= scatter_blocks_per_dev * dst_dev_idx;
    } else {
        int scatter_blocks_per_dev = G.src.cols() / globals::COL_BLOCK_SIZE / globals::NUM_DEVICES;
        dst_dev_idx = indices.w / scatter_blocks_per_dev;
        indices.w -= scatter_blocks_per_dev * dst_dev_idx;
    }
    if constexpr (GATHER_AXIS == 0) {
        int gather_blocks_per_dev = G.dst.batch() / globals::NUM_DEVICES;
        indices.x += gather_blocks_per_dev * G.dev_idx;
    } else if constexpr (GATHER_AXIS == 1) {
        int gather_blocks_per_dev = G.dst.depth() / globals::NUM_DEVICES;
        indices.y += gather_blocks_per_dev * G.dev_idx;
    } else if constexpr (GATHER_AXIS == 2) {
        int gather_blocks_per_dev = G.dst.rows() / globals::ROW_BLOCK_SIZE / globals::NUM_DEVICES;
        indices.z += gather_blocks_per_dev * G.dev_idx;
    } else {
        int gather_blocks_per_dev = G.dst.cols() / globals::COL_BLOCK_SIZE / globals::NUM_DEVICES;
        indices.w += gather_blocks_per_dev * G.dev_idx;
    }

    // Wait for inputs to be arrived
    wait(inputs_arrived, 0);
    asm volatile("{fence.proxy.async.shared::cta;}" ::: "memory"); // make writes to smem visible

    // Store data to destination device
    asm volatile("{cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3, %4}], [%5];}"
        :: "l"(G.dst[dst_dev_idx].template get_tma<globals::tile, 2>()),
            "r"(indices.w * globals::COL_BLOCK_SIZE), "r"(indices.z * globals::ROW_BLOCK_SIZE), "r"(indices.y), "r"(indices.x),
            "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&tile[0])))
        : "memory");
}

template <int SCATTER_AXIS = 2, int GATHER_AXIS = 1>
void entrypoint(kittens::py::TKParallelTensor &dst, kittens::py::TKParallelTensor &src) {
    static_assert(SCATTER_AXIS < 4 && GATHER_AXIS < 4, "Scatter and gather axes must be between 0 and 3");
    static_assert(SCATTER_AXIS != GATHER_AXIS, "Scatter and gather axes must be different");

    kittens::py::parallel_tensor_check(dst, src);

    int actual_scatter_axis = (4 - src.data_.dim()) + SCATTER_AXIS;
    int actual_gather_axis = (4 - src.data_.dim()) + GATHER_AXIS;
    TORCH_CHECK(actual_gather_axis >= 0, "actual_gather_axis is less than 0");
    TORCH_CHECK(actual_scatter_axis >= 0, "actual_scatter_axis is less than 0");

    TORCH_CHECK(src.data_.dim() == dst.data_.dim(), "src and dst must have the same number of dimensions");

    for (int i = 0; i < src.data_.dim(); ++i) {
        if (i == actual_scatter_axis) {
            if constexpr (SCATTER_AXIS < 2) {
                TORCH_CHECK(src.data_.size(i) % src.local_world_size_ == 0, "src must be divisible by the local world size for the scatter axis");
            } else if constexpr (SCATTER_AXIS == 2) {
                TORCH_CHECK(src.data_.size(i) % (src.local_world_size_ * globals::ROW_BLOCK_SIZE) == 0, "src must be divisible by the local world size times row block size for the scatter axis");
            } else if constexpr (SCATTER_AXIS == 3) {
                TORCH_CHECK(src.data_.size(i) % (src.local_world_size_ * globals::COL_BLOCK_SIZE) == 0, "src must be divisible by the local world size times col block size for the scatter axis");
            }
            TORCH_CHECK(src.data_.size(i) / src.local_world_size_ == dst.data_.size(i), "dst scatter dimension must be src scatter dimension divided by the local world size");
        } else if (i == actual_gather_axis) {
            if constexpr (GATHER_AXIS < 2) {
                TORCH_CHECK(dst.data_.size(i) % dst.local_world_size_ == 0, "dst must be divisible by the local world size for the gather axis");
            } else if constexpr (GATHER_AXIS == 2) {
                TORCH_CHECK(dst.data_.size(i) % (dst.local_world_size_ * globals::ROW_BLOCK_SIZE) == 0, "dst must be divisible by the local world size times row block size for the gather axis");
            } else if constexpr (GATHER_AXIS == 3) {
                TORCH_CHECK(dst.data_.size(i) % (dst.local_world_size_ * globals::COL_BLOCK_SIZE) == 0, "dst must be divisible by the local world size times col block size for the gather axis");
            }
            TORCH_CHECK(dst.data_.size(i) / dst.local_world_size_ == src.data_.size(i), "src gather dimension must be dst gather dimension divided by the local world size");
        } else {
            TORCH_CHECK(src.data_.size(i) == dst.data_.size(i), "src and dst must have the same size for all dimensions except the scatter and gather axes");
        }
    }

    // Instantiate globals
    globals G {
        .src = kittens::py::parallel_tensor_to_pgl<typename globals::parallel_layout>(src),
        .dst = kittens::py::parallel_tensor_to_pgl<typename globals::parallel_layout>(dst),
        .dev_idx = src.local_rank_
    };

    // Run kernel
    kittens::py::launch_kernel<config, globals, kernel<2, 1>>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("all2all_s2g1", &entrypoint<2, 1>);
    m.def("all2all_s1g2", &entrypoint<1, 2>);
}
