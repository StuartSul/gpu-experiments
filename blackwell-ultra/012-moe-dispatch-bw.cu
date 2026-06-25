#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 6;
    static constexpr int NUM_THREADS = 1;
};

struct globals {
    static constexpr int NUM_DEVICES = 4;
    static constexpr int ROW_BLOCK_SIZE = 128;
    static constexpr int COL_BLOCK_SIZE = 128;

    using tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE>;
    using send_gl = gl<bf16, 1, -1, -1, -1, tile>;
    using recv_gl = pgl<gl<bf16, 1, -1, -1, -1, tile>, NUM_DEVICES, false>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    send_gl send;
    recv_gl recv;
    index_gl dst_rank;
    index_gl dst_block;

    __host__ inline dim3 grid() const {
        return dim3(send.cols() / COL_BLOCK_SIZE, send.rows() / ROW_BLOCK_SIZE, 1);
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(tile) + 1024);
    }
};

__device__ inline void dispatch_kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::tile &tile = allocator.allocate<globals::tile>();

    const int src_block_idx = blockIdx.y;
    const int col_idx = blockIdx.x;
    const int dst_rank = G.dst_rank[{src_block_idx}];
    const int dst_block_idx = G.dst_block[{src_block_idx}];

    if (dst_rank < 0) return; // this source-block slot is past the dispatch schedule

    __shared__ semaphore inputs_arrived;
    init_semaphore(inputs_arrived, 0, 1);
    tma::expect_bytes(inputs_arrived, sizeof(globals::tile));
    tma::load_async(tile, G.send, {src_block_idx, col_idx}, inputs_arrived);
    wait(inputs_arrived, 0);
    tma::store_async(G.recv[dst_rank], tile, {dst_block_idx, col_idx});
}

void dispatch(
    const at::Tensor &send,
    const at::Tensor &recv,
    const std::vector<int64_t> &recv_ptrs,
    const at::Tensor &dst_rank,
    const at::Tensor &dst_block
) {
    bf16 *recv_data[globals::NUM_DEVICES];
    for (int i = 0; i < globals::NUM_DEVICES; ++i)
        recv_data[i] = reinterpret_cast<bf16*>(recv_ptrs[i]);

    globals G {
        .send = kittens::py::tensor_to_gl<globals::send_gl>(send),
        .recv = globals::recv_gl{recv_data, nullptr, static_cast<size_t>(1), static_cast<size_t>(recv.size(0)), static_cast<size_t>(recv.size(1))},
        .dst_rank = kittens::py::tensor_to_gl<globals::index_gl>(dst_rank),
        .dst_block = kittens::py::tensor_to_gl<globals::index_gl>(dst_block),
    };

    kittens::py::launch_kernel<config, globals, dispatch_kernel>(G);
}

PYBIND11_MODULE(_C, m) {
    m.def("dispatch", &dispatch, "MoE reorganize-and-dispatch (TMA 128x128 tile push over NVLink)",
          pybind11::arg("send"), pybind11::arg("recv"), pybind11::arg("recv_ptrs"), pybind11::arg("dst_rank"), pybind11::arg("dst_block"));
}
