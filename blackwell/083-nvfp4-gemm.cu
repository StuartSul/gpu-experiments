/*

Reference: https://arxiv.org/pdf/2509.25149

NVFP4 uses 2-level scaling:

    FP4 (S1E2M1) --> FP8 (S1E4M3) --> FP32 (IEEE)
    (-6 ~ 6)         (-448 ~ 448)

NVFP4 quantization steps:

    Unlike MXFP8 quantization, this usually requires 2-pass through the HBM. Once for
    global scale (FP32) and once for per-block scale (FP8).

    Note that global scale does NOT have to be truly global; you can customize and
    use it over a row or an arbitrary-sized block.

    1. Compute global tensor-level scale: s_global_enc = 6 * 448 / max(|x_i|)
                                          s_global_dec = 1 / s_global_enc
    2. Compute local block-level scale:   s_local_dec = fp8e4m3_round_to_even(s_global_enc * max(|x_j|) / 6)
                                          s_local_enc = s_global_enc / fp32(s_local_dec)

    Note that the goal is to have s_local_enc * s_local_dec * s_global_dec ~ 1.
    That is, we **encode only using s_local_enc**, but decode in two steps (once by the tensor cores, once by ourselves)

    3. x_q = fp4(x_i * s_local_enc)

NVFP4 GEMM steps:

    1. Perform tensor-core GEMM
    2. Perform CUDA-core decode with s_global_dec

Nvidia also recommends the following:

    - 16x16 block scaling on weights for scaling consistency between Fprop/Bprop
    - The first couple layers kept in BF16 / MXFP8
    - The last few layers (fewer than 15%) kept in BF16 / MXFP8
    - Embeddings, norms, attns kept in original precision (BF16 or FP32)
    - Optimizer, weights, weight gradients kept in FP32
    - Tensor parallel reductions in BF16
    - Random Hadamard Transformation (RHT) applied to both operands in Wgrad (transpose - RHT - NVFP4 quant - GEMM) (but not anywhere else)
    - Stochastic rounding applied during NVFP4 quantization of activation gradient for Dgrad
    - Towards the end of training, switch everything back to BF16

*/
#include <kittens.cuh>
#include <prototype.cuh>
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int NUM_BLOCKS = 148;

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 2; 
    static constexpr int PRODUCER_WARPGROUPS = 1; 
    static constexpr int NUM_WARPGROUPS      = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS; 
    static constexpr int NUM_WARPS           = NUM_WARPGROUPS * WARPGROUP_WARPS; 
    static constexpr int NUM_THREADS         = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 32;
    static constexpr int CONSUMER_REGISTERS = 160;
};

struct globals {
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int REDUCTION_BLOCK = 128;

    using A_fp4_tile = st_fp4_2<ROW_BLOCK / 2, REDUCTION_BLOCK / 2>; // CTA-distributed & 2-packed
    using A_local_scale_tile = st_fp8e4m3<ROW_BLOCK / 2, REDUCTION_BLOCK>;
    using B_fp4_tile = st_fp4_2<COL_BLOCK / 2, REDUCTION_BLOCK>;
    using B_local_scale_tile = st_fp8e4m3<COL_BLOCK / 2, REDUCTION_BLOCK>;
    using C_tile = st_bf<ROW_BLOCK / 2, COL_BLOCK / 2>;

    using A_fp4_gl = gl<fp4_2, 1, 1, -1, -1, A_fp4_tile>;
    using A_local_scale_gl = gl<fp8e4m3, 1, -1, -1, -1, A_local_scale_tile>;
    using A_global_scale_gl = gl<float, 1, 1, 1, 1>;
    using B_fp4_gl = gl<fp4_2, 1, 1, -1, -1, B_fp4_tile>;
    using B_local_scale_gl = gl<fp8e4m3, 1, -1, -1, -1, B_local_scale_tile>;
    using B_global_scale_gl = gl<float, 1, 1, 1, 1>;
    using C_gl = gl<bf16, 1, 1, -1, -1, C_tile>;

    A_fp4_gl A_fp4;
    A_local_scale_gl A_local_scale;
    A_global_scale_gl A_global_scale;
    B_fp4_gl B_fp4;
    B_local_scale_gl B_local_scale;
    B_global_scale_gl B_global_scale;
    C_gl C;
};

__device__ inline void kernel(const globals &G) {
    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    globals::A_fp4_tile &A_fp4_tile = allocator.allocate<globals::A_fp4_tile>();

    // Set up mbarriers
    __shared__ semaphore inputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
    }

    // Test
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        tma::expect_bytes(inputs_arrived, sizeof(globals::A_fp4_tile));
        tma::load_async(A_fp4_tile, G.A_fp4, {0, 0}, inputs_arrived);
        wait(inputs_arrived, 0);

        for (int i = 0; i < A_fp4_tile.rows; i++) {
            for (int j = 0; j < A_fp4_tile.cols; j++) {
                fp4_2 tmp = *A_fp4_tile.idx(A_fp4_tile.data, {i, j});
                // static constexpr int swizzle_repeat = A_fp4_tile.swizzle_bytes * 8;
                // static constexpr int subtile_cols   = A_fp4_tile.swizzle_bytes / sizeof(fp4);
                // const int outer_idx = c/subtile_cols;
                // const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
                // const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
                // return (T*)(addr ^ swizzle);
                printf("%u ", tmp.__x);
            }
            printf("\n");
        }
    }
}

void entrypoint(
    const at::Tensor &A_fp4,
    const at::Tensor &A_local_scale,
    const at::Tensor &A_global_scale,
    const at::Tensor &B_fp4,
    const at::Tensor &B_local_scale,
    const at::Tensor &B_global_scale,
    at::Tensor &C
) {
    globals G {
        .A_fp4 = kittens::py::tensor_to_gl<typename globals::A_fp4_gl>(A_fp4),
        .A_local_scale = kittens::py::tensor_to_gl<typename globals::A_local_scale_gl>(A_local_scale),
        .A_global_scale = kittens::py::tensor_to_gl<typename globals::A_global_scale_gl>(A_global_scale),
        .B_fp4 = kittens::py::tensor_to_gl<typename globals::B_fp4_gl>(B_fp4),
        .B_local_scale = kittens::py::tensor_to_gl<typename globals::B_local_scale_gl>(B_local_scale),
        .B_global_scale = kittens::py::tensor_to_gl<typename globals::B_global_scale_gl>(B_global_scale),
        .C = kittens::py::tensor_to_gl<typename globals::C_gl>(C),
    };
    kittens::py::launch_kernel<config, globals, kernel>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    m.def("nvfp4_gemm", &entrypoint);
}
