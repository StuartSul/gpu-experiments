#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;
    device<NUM_DEVICES>::barrier_t barriers;
    const int dev_idx;
};

__device__ inline void kernel(const globals &G) {
    device<globals::NUM_DEVICES>::sync_on_exit(G.barriers, G.dev_idx);
}

void entrypoint(kittens::py::TKParallelTensor &barriers) {
    // Instantiate globals
    globals G {
        .barriers = kittens::py::parallel_tensor_to_pgl<device<globals::NUM_DEVICES>::barrier_t>(barriers),
        .dev_idx = barriers.local_rank_
    };

    // Run kernel
    kittens::py::launch_kernel<config, globals, kernel>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("sync", &entrypoint);
}
