#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

// ThunderKittens macro check
#if !defined(KITTENS_HOPPER) || !defined(KITTENS_BLACKWELL)
    #error "KITTENS_HOPPER and KITTENS_BLACKWELL macros must be defined for Blackwell compilation"
#endif

// Kernel configuration
struct config {
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 1;
    static constexpr int PRODUCER_WARPGROUPS = 1;
    static constexpr int NUM_WARPGROUPS = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
    static constexpr int NUM_WARPS = NUM_WARPGROUPS * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;
};

// Kernel globals
struct globals {
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 128;
    static constexpr int REDUCTION_BLOCK = 128;

    using A_tile = st_fp8e4m3<ROW_BLOCK, REDUCTION_BLOCK>;
    using B_tile = st_fp8e4m3<COL_BLOCK, REDUCTION_BLOCK>;
    using C_tile = st_fl<ROW_BLOCK, COL_BLOCK>;

    gl<fp8e4m3, 1, 1, -1, -1, A_tile> A;
    gl<fp8e4m3, 1, 1, -1, -1, B_tile> B;
    gl<float, 1, 1, -1, -1, C_tile> C;

    __host__ inline dim3 grid() { return dim3(1); } // use single block
    __host__ inline dim3 block() { return dim3(config::NUM_THREADS); }
    __host__ inline int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }

    struct pipeline_inputs {
        A_tile A;
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };
};

// Kernel implementation
__global__ __launch_bounds__(config::NUM_THREADS, 1)
void fp8_matmul_kernel(const __grid_constant__ globals G) {
    // Shared memory declaration
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    // Warpgroup configuration
    int lane_id = warp::laneid();
    int warp_id = warpgroup::warpid();
    int warpgroup_id = warpgroup::groupid();

    // Allocate shared memory
    static_assert(sizeof(globals::pipeline_inputs) + sizeof(globals::pipeline_outputs) <= config::DYNAMIC_SHARED_MEMORY);
    globals::pipeline_inputs &inputs = allocator.allocate<globals::pipeline_inputs>();
    globals::pipeline_outputs &outputs = allocator.allocate<globals::pipeline_outputs>();

    // Allocate tensor memory
    tensor_allocator<1, 1> tm_allocator {};
    using tm_t = tt<float, globals::ROW_BLOCK, globals::COL_BLOCK>;
    tm_t tm = tm_allocator.template allocate<tm_t>(0);

    // Set up mbarriers
    __shared__ semaphore inputs_arrived;
    __shared__ semaphore outputs_arrived;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(outputs_arrived, 0, 1);
    }
    __syncthreads();

    // Main divergence
    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        // Producer group
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        if (warp_id == 3 && lane_id == 0) {
            // Load input matrices
            tma::expect_bytes(inputs_arrived, sizeof(globals::pipeline_inputs));
            tma::load_async(inputs.A, G.A, {0, 0}, inputs_arrived);
            tma::load_async(inputs.B, G.B, {0, 0}, inputs_arrived);
        } else if (warp_id == 0 && lane_id == 0) {
            // Launch tensor core matrix multiply
            wait(inputs_arrived, 0);
            mm_ABt(tm, inputs.A, inputs.B, outputs_arrived);
        }
    } else if (warpgroup_id == 0) {
        // Consumer group
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        // Wait for the matmul to complete
        wait(outputs_arrived, 0);

        // Load the output from tensor memory into registers
        rt_fl<globals::ROW_BLOCK / 4, globals::COL_BLOCK> C_reg;
        warpgroup::load_async(C_reg, tm);
        tensor_load_wait();

        // Store back to global memory
        warpgroup::store(outputs.C, C_reg);
        warpgroup::sync(0);
        warpgroup::tma::store_async(G.C, outputs.C, {0, 0});
        warpgroup::tma::store_async_read_wait();
        warpgroup::sync(0);
    }
}

// Python bindings
PYBIND11_MODULE(_C, m) {
    m.doc() = "";
    kittens::py::bind_kernel<fp8_matmul_kernel>(m, "fp8_matmul",
        &globals::A,
        &globals::B,
        &globals::C
    );
}
