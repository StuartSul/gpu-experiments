/*
    Further optimizations experiments on multimem all reduce
    
    Multimem ld_reduce + st (256 threads) : 1.124 ms | 835.95 GB/s
    Multimem ld_reduce + st (128 threads) : 1.124 ms | 835.83 GB/s
    Multimem ld_reduce + TMA vector store : 1.125 ms | 835.05 GB/s  <-- with additional intermediate SMEM, but nearly same speed
    TMA vector load + store_add : X
    TMA tile load + store_add (swizzle ON) : X
    TMA tile load + store_add (swizzle OFF) : X

    Can't do store_add since it requires exclusion of itself, and little benchmark does show it would be slower (with 8x more TMA per device)
*/

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;

namespace all_reduce {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int MIN_BLOCKS_PER_SM = 8;
    static constexpr int NUM_WARPGROUPS = 1;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    // static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    static constexpr int NUM_ELEMS_PER_INST = 2;
    static constexpr int NUM_ELEMS_PER_BLOCK = config::NUM_THREADS * NUM_ELEMS_PER_INST;
    
    using shared_vec = sv_bf<NUM_ELEMS_PER_BLOCK>;
    using parallel_layout = pgl<gl<bf16, -1, -1, -1, -1>, NUM_DEVICES, true, shared_vec>;

    parallel_layout tensor;
    const int dev_idx;

    __host__ inline dim3 grid() const {
        return dim3(tensor.numel() / NUM_ELEMS_PER_BLOCK / NUM_DEVICES);
    }

    __host__ inline int dynamic_shared_memory() const {
        return sizeof(shared_vec) + 1024;
    }
};

__device__ inline void kernel(const globals &G) {

    { 
        // const size_t N_total = G.tensor.numel();
        // const size_t N_per_dev = N_total / globals::NUM_DEVICES;
        // const size_t base_idx = N_per_dev * G.dev_idx + globals::NUM_ELEMS_PER_BLOCK * blockIdx.x;
        // const size_t idx = base_idx + globals::NUM_ELEMS_PER_INST * threadIdx.x;

        // bf16_2 tmp;
        // multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<bf16_2 *>(&G.tensor.mc_ptr[idx]));
        // multimem<bf16_2>::st(reinterpret_cast<bf16_2 *>(&G.tensor.mc_ptr[idx]), tmp);
    }
    
    {
        const size_t N_total = G.tensor.numel();
        const size_t N_per_dev = N_total / globals::NUM_DEVICES;
        const size_t base_idx = N_per_dev * G.dev_idx + globals::NUM_ELEMS_PER_BLOCK * blockIdx.x;
        const size_t idx = base_idx + globals::NUM_ELEMS_PER_INST * threadIdx.x;

        extern __shared__ int __shm[];
        tma_swizzle_allocator allocator((int*)&__shm[0]);
        globals::shared_vec &vec = allocator.allocate<globals::shared_vec>();

        bf16_2 tmp;
        multimem<bf16_2>::ld_reduce<reduce_op::ADD>(tmp, reinterpret_cast<bf16_2 *>(&G.tensor.mc_ptr[idx]));

        uint32_t shared_ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&vec.data[globals::NUM_ELEMS_PER_INST * threadIdx.x]));
        move<bf16_2>::sts(shared_ptr, tmp);
        __syncthreads();
    
        if (threadIdx.x == 0) {
            tma::store_async(G.tensor, vec,
                {0, 0, base_idx / G.tensor.cols(), (base_idx % G.tensor.cols()) / globals::NUM_ELEMS_PER_BLOCK});
        }
    }
}

} // namespace all_reduce

namespace all_reduce_barrier {

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    device<NUM_DEVICES>::barrier_t barrier;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    device<globals::NUM_DEVICES>::sync_on_exit(G.barrier, G.dev_idx);
}

} // namespace all_reduce_barrier

void entrypoint(
    kittens::py::TKParallelTensor &tensor,
    kittens::py::TKParallelTensor &barrier
) {
    kittens::py::parallel_tensor_check(tensor, barrier);

    TORCH_CHECK(tensor.data_.numel() % (all_reduce::globals::NUM_DEVICES * all_reduce::globals::NUM_ELEMS_PER_BLOCK) == 0, 
        "The total number of tensor elements must be divisible by NUM_DEVICES * NUM_ELEMS_PER_BLOCK");

    all_reduce::globals all_reduce_G {
        .tensor = kittens::py::parallel_tensor_to_pgl<typename all_reduce::globals::parallel_layout>(tensor),
        .dev_idx = tensor.local_rank_
    };

    kittens::py::launch_kernel<all_reduce::config, all_reduce::globals, all_reduce::kernel>(all_reduce_G);

    all_reduce_barrier::globals barrier_G {
        .barrier = kittens::py::parallel_tensor_to_pgl<device<all_reduce_barrier::globals::NUM_DEVICES>::barrier_t>(barrier),
        .dev_idx = barrier.local_rank_
    };

    kittens::py::launch_kernel<all_reduce_barrier::config, all_reduce_barrier::globals, all_reduce_barrier::kernel>(barrier_G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_all_reduce", &entrypoint);
}
