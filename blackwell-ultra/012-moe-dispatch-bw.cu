#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 6;
    static constexpr int NUM_THREADS = 128;
};

struct globals {
    static constexpr int NUM_DEVICES = 4;
    static constexpr int ROW_BLOCK_SIZE = 128;
    static constexpr int COL_BLOCK_SIZE = 128;

    using token_tile = st_bf<ROW_BLOCK_SIZE, COL_BLOCK_SIZE, false>;
    using token_vec = sv_bf<COL_BLOCK_SIZE>;

    using tokens_gl = gl<bf16, 1, 1, -1, -1, token_vec>;
    using recv_buffer_gl = pgl<gl<bf16, 1, -1, -1, -1, token_tile>, NUM_DEVICES, false>;
    using index_gl = gl<int, 1, 1, 1, -1>;

    tokens_gl tokens;
    recv_buffer_gl recv_buffer;
    index_gl schedule_src_token_idx;
    index_gl schedule_dst_rank;
    index_gl schedule_dst_block_idx;

    __host__ inline dim3 grid() const {
        return dim3(tokens.cols() / COL_BLOCK_SIZE, schedule_dst_rank.cols());
    }
    __host__ inline int dynamic_shared_memory() const {
        return static_cast<int>(sizeof(token_tile) + 1024);
    }
};

__device__ inline void dispatch_kernel(const globals &G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::token_tile &token_tile = allocator.allocate<globals::token_tile>();

    const int tid = threadIdx.x;
    const int col_idx = blockIdx.x;
    const int src_block_idx = blockIdx.y;
    const int dst_rank = G.schedule_dst_rank[{src_block_idx}];
    const int dst_block_idx = G.schedule_dst_block_idx[{src_block_idx}];

    if (dst_rank < 0) return; // this source-block slot is past the dispatch schedule

    const int src_token_idx = G.schedule_src_token_idx[{src_block_idx * globals::ROW_BLOCK_SIZE + tid}];
    globals::token_vec &dst_vec = *reinterpret_cast<globals::token_vec*>(&token_tile.data[tid * globals::COL_BLOCK_SIZE]);

    __shared__ semaphore inputs_arrived;
    const int num_valid = __syncthreads_count(src_token_idx >= 0);
    if (tid == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        tma::expect_bytes(inputs_arrived, num_valid * sizeof(globals::token_vec));
    }
    __syncthreads();

    if (src_token_idx >= 0) {
        tma::load_async(dst_vec, G.tokens, {src_token_idx, col_idx}, inputs_arrived); // causes peeling loop but still faster
    } else {
        #pragma unroll
        for (int j = 0; j < globals::COL_BLOCK_SIZE; ++j)
            dst_vec[j] = __float2bfloat16(0.0f);
    }

    __syncthreads(); // for the padding writes
    wait(inputs_arrived, 0);

    if (tid == 0) {
        tma::store_async(G.recv_buffer[dst_rank], token_tile, {dst_block_idx, col_idx});
        tma::store_async_wait();
    }
}

void dispatch(
    const at::Tensor &tokens,
    const at::Tensor &recv_buffer,
    const std::vector<int64_t> &recv_buffer_ptrs,
    const at::Tensor &schedule_src_token_idx,
    const at::Tensor &schedule_dst_rank,
    const at::Tensor &schedule_dst_block_idx
) {
    bf16 *recv_buffer_data[globals::NUM_DEVICES];
    for (int i = 0; i < globals::NUM_DEVICES; ++i)
        recv_buffer_data[i] = reinterpret_cast<bf16*>(recv_buffer_ptrs[i]);

    globals G {
        .tokens = kittens::py::tensor_to_gl<globals::tokens_gl>(tokens),
        .recv_buffer = globals::recv_buffer_gl{recv_buffer_data, nullptr, static_cast<size_t>(1), static_cast<size_t>(recv_buffer.size(0)), static_cast<size_t>(recv_buffer.size(1))},
        .schedule_src_token_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_src_token_idx),
        .schedule_dst_rank = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_rank),
        .schedule_dst_block_idx = kittens::py::tensor_to_gl<globals::index_gl>(schedule_dst_block_idx),
    };

    kittens::py::launch_kernel<config, globals, dispatch_kernel>(G);
}

PYBIND11_MODULE(_C, m) {
    m.def("dispatch", &dispatch, "MoE gather-and-dispatch (TMA vector gather -> single 128x128 token_tile push over NVLink)",
          pybind11::arg("tokens"), pybind11::arg("recv_buffer"), pybind11::arg("recv_buffer_ptrs"), 
          pybind11::arg("schedule_src_token_idx"), pybind11::arg("schedule_dst_rank"), pybind11::arg("schedule_dst_block_idx"));
}
