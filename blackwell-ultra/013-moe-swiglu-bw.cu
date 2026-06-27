#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

#include <ATen/ops/empty_like.h>

using namespace kittens;

namespace moe_swigluer {

struct config {
    static constexpr int CLUSTER_SIZE = 2;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int LOAD_PIPE_DEPTH = 5;
    static constexpr int EPI_PIPE_DEPTH = 8;
    static constexpr bool OVERLAP_EPI = true;

    static constexpr int SUPERGROUP_SIZE = 12;
    static constexpr int Mb = 256; // must not change
    static constexpr int Nb = 256; // must not change
    static constexpr int Kb = 128; // must not change
    static constexpr int B_SC_SIZE = Nb/128;
    static constexpr int NUM_D_TILES = 2;

    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;
};

template <typename C>
struct globals {
    using x_tile      = st_bf<C::Mb / 2, C::Kb>;
    using weight_tile = st_bf<C::Nb / 2, C::Kb>;

    using activation_gl = gl<bf16, 1, 1, -1, -1, x_tile>;
    using weight_gl     = gl<bf16, 1, -1, -1, -1, weight_tile>;
    using index_gl      = gl<int, 1, 1, 1, -1>;

    activation_gl x;  // (total_tokens, H)
    weight_gl w_gate; // (E, H, I)
    weight_gl w_up;   // (E, H, I)
    weight_gl w_down; // (E, I, H)
    index_gl tokens_per_expert; // (E,)
    activation_gl y;  // (total_tokens, H)

    __host__ __inline__ dim3 grid() const { return dim3(C::CLUSTER_SIZE); } // TODO
};

template <typename C>
__device__ inline void swiglu_kernel(const globals<C> &g) {
    // TODO
}

at::Tensor swiglu(
    const at::Tensor &x,
    const at::Tensor &w_gate,
    const at::Tensor &w_up,
    const at::Tensor &w_down,
    const at::Tensor &tokens_per_expert
) {
    using C = config;
    using G = globals<C>;

    at::Tensor y = at::empty_like(x);

    G g {
        .x = kittens::py::tensor_to_gl<G::activation_gl>(x),
        .w_gate = kittens::py::tensor_to_gl<G::weight_gl>(w_gate),
        .w_up = kittens::py::tensor_to_gl<G::weight_gl>(w_up),
        .w_down = kittens::py::tensor_to_gl<G::weight_gl>(w_down),
        .tokens_per_expert = kittens::py::tensor_to_gl<G::index_gl>(tokens_per_expert),
        .y = kittens::py::tensor_to_gl<G::activation_gl>(y),
    };

    kittens::py::launch_kernel<C, G, swiglu_kernel<C>>(g);

    return y;
}

} // namespace swigluer

PYBIND11_MODULE(_C, m) {
    m.def("swiglu", &swigluer::swiglu, "MoE Swiglu",
          pybind11::arg("x"), pybind11::arg("w_gate"), pybind11::arg("w_up"), pybind11::arg("w_down"),
          pybind11::arg("tokens_per_expert"));
}
