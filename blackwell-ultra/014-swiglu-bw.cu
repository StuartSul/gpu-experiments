#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace swigluer {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_THREADS = 256;
    static constexpr int NUM_WARPS = NUM_THREADS / WARP_THREADS;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
};

struct globals {
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 128;

    using activation_tile = st_bf<ROW_BLOCK, COL_BLOCK>;
    using activation_gl = gl<bf16, 1, 1, -1, -1, activation_tile>;

    activation_gl a;
    activation_gl b;
    activation_gl c;

    __host__ inline dim3 grid() const {
        const int num_tiles = (c.rows() / ROW_BLOCK) * (c.cols() / COL_BLOCK);
        return dim3((num_tiles + 2) / 3); // round up
    }
};

__device__ inline void swiglu_kernel(const globals &g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    globals::activation_tile (&a_smem)[3] = allocator.allocate<globals::activation_tile, 3>();
    globals::activation_tile (&b_smem)[3] = allocator.allocate<globals::activation_tile, 3>();

    const int row_blocks = g.c.rows() / globals::ROW_BLOCK;
    const int col_blocks = g.c.cols() / globals::COL_BLOCK;
    const int num_tiles = col_blocks * row_blocks;
    const int first_tile_idx = blockIdx.x * 3;

    __shared__ semaphore inputs_arrived[3];
    int first_row, first_col;
    if (threadIdx.x == 0) {
        first_row = first_tile_idx / col_blocks;
        first_col = first_tile_idx % col_blocks;
        #pragma unroll
        for (int stage = 0; stage < 3; ++stage) {
            init_semaphore(inputs_arrived[stage], 0, 1);
            const int tile_idx = first_tile_idx + stage;
            if (tile_idx < num_tiles) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::expect_bytes(inputs_arrived[stage], sizeof(a_smem[stage]) + sizeof(b_smem[stage]));
                tma::load_async(a_smem[stage], g.a, {row, col}, inputs_arrived[stage]);
                tma::load_async(b_smem[stage], g.b, {row, col}, inputs_arrived[stage]);
            }
        }
    }
    __syncthreads();

    using compute_group = group<config::NUM_WARPS>;
    rt_fl<globals::ROW_BLOCK / config::NUM_WARPS, globals::COL_BLOCK> a, b, denominator;

    #pragma unroll
    for (int stage = 0; stage < 3; ++stage) {
        const int tile_idx = first_tile_idx + stage;
        if (tile_idx < num_tiles) {
            wait(inputs_arrived[stage], 0);
            compute_group::load(a, a_smem[stage]);
            compute_group::load(b, b_smem[stage]);
            compute_group::mul(denominator, a, -1.4426950408889634f);
            compute_group::exp2(denominator, denominator);
            compute_group::add(denominator, denominator, 1.0f);
            compute_group::div(a, a, denominator);
            compute_group::mul(a, a, b);
            compute_group::store(a_smem[stage], a);
            __syncthreads();
            if (threadIdx.x == 0) {
                // This improves throughput
                int row = first_row;
                int col = first_col + stage;
                if (col >= col_blocks) {
                    ++row;
                    col -= col_blocks;
                }
                tma::store_async(g.c, a_smem[stage], {row, col});
            }
        }
    }
}

void swiglu(const at::Tensor &a, const at::Tensor &b, at::Tensor &c) {
    globals g {
        .a = kittens::py::tensor_to_gl<globals::activation_gl>(a),
        .b = kittens::py::tensor_to_gl<globals::activation_gl>(b),
        .c = kittens::py::tensor_to_gl<globals::activation_gl>(c),
    };

    kittens::py::launch_kernel<config, globals, swiglu_kernel>(g);
}

} // namespace swigluer

PYBIND11_MODULE(_C, m) {
    m.def("swiglu", &swigluer::swiglu,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"));
}
