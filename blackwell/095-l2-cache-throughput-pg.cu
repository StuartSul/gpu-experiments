/*
    B200 Results:
        L2 Size=126.50 MB
        ==================================
        N=2048, REPEAT=128
        Size: 8 MB
        Time: 57.42 us
        Throughput: 18698.17 GB/s
        ==================================
        N=3072, REPEAT=128
        Size: 18 MB
        Time: 126.98 us
        Throughput: 19025.33 GB/s
        ==================================
        N=4096, REPEAT=128
        Size: 32 MB
        Time: 222.86 us
        Throughput: 19272.33 GB/s
        ==================================
        N=4224, REPEAT=128
        Size: 34 MB
        Time: 238.71 us
        Throughput: 19134.60 GB/s
        ==================================
        N=4352, REPEAT=128
        Size: 36 MB
        Time: 253.02 us
        Throughput: 19163.01 GB/s
        ==================================
        N=4480, REPEAT=128
        Size: 38 MB
        Time: 265.90 us
        Throughput: 19322.96 GB/s
        ==================================
        N=4608, REPEAT=128
        Size: 40 MB
        Time: 282.21 us
        Throughput: 19261.48 GB/s
        ==================================
        N=4736, REPEAT=128
        Size: 43 MB
        Time: 302.50 us
        Throughput: 18981.67 GB/s
        ==================================
        N=4864, REPEAT=128
        Size: 45 MB
        Time: 313.42 us
        Throughput: 19324.36 GB/s
        ==================================
        N=4992, REPEAT=128
        Size: 48 MB
        Time: 330.36 us
        Throughput: 19310.86 GB/s
        ==================================
        N=5120, REPEAT=128
        Size: 50 MB
        Time: 356.51 us
        Throughput: 18823.83 GB/s
        ==================================
        N=5248, REPEAT=128
        Size: 53 MB
        Time: 366.81 us
        Throughput: 19221.23 GB/s
        ==================================
        N=5376, REPEAT=128
        Size: 55 MB
        Time: 381.17 us
        Throughput: 19410.41 GB/s
        ==================================
        N=5504, REPEAT=128
        Size: 58 MB
        Time: 404.28 us
        Throughput: 19182.72 GB/s
        ==================================
        N=5632, REPEAT=128
        Size: 60 MB
        Time: 419.69 us
        Throughput: 19348.00 GB/s
        ==================================
        N=5760, REPEAT=128
        Size: 63 MB
        Time: 439.42 us
        Throughput: 19328.72 GB/s
        ==================================
        N=5888, REPEAT=128
        Size: 66 MB
        Time: 461.12 us
        Throughput: 19246.84 GB/s
        ==================================
        N=6016, REPEAT=128
        Size: 69 MB
        Time: 475.04 us
        Throughput: 19503.92 GB/s
        ==================================
        N=6144, REPEAT=128
        Size: 72 MB
        Time: 501.56 us
        Throughput: 19267.08 GB/s
        ==================================
        N=6272, REPEAT=128
        Size: 75 MB
        Time: 525.23 us
        Throughput: 19173.64 GB/s
        ==================================
        N=6400, REPEAT=128
        Size: 78 MB
        Time: 555.78 us
        Throughput: 18866.75 GB/s
        ==================================
        N=6528, REPEAT=128
        Size: 81 MB
        Time: 583.52 us
        Throughput: 18695.86 GB/s
        ==================================
        N=6656, REPEAT=128
        Size: 84 MB
        Time: 637.61 us
        Throughput: 17787.46 GB/s
        ==================================
        N=6784, REPEAT=128
        Size: 88 MB
        Time: 605.14 us
        Throughput: 19469.45 GB/s
        ==================================
        N=6912, REPEAT=128
        Size: 91 MB
        Time: 697.41 us
        Throughput: 17537.25 GB/s
        ==================================
        N=7040, REPEAT=128
        Size: 95 MB
        Time: 738.64 us
        Throughput: 17177.20 GB/s
        ==================================
        N=7168, REPEAT=128
        Size: 98 MB
        Time: 788.42 us
        Throughput: 16683.20 GB/s
        ==================================
        N=7296, REPEAT=128
        Size: 102 MB
        Time: 692.74 us
        Throughput: 19671.48 GB/s
        ==================================
        N=7424, REPEAT=128
        Size: 105 MB
        Time: 899.08 us
        Throughput: 15693.39 GB/s
        ==================================
        N=7552, REPEAT=128
        Size: 109 MB
        Time: 863.00 us
        Throughput: 16918.17 GB/s
        ==================================
        N=7680, REPEAT=128
        Size: 112 MB
        Time: 978.72 us
        Throughput: 15427.73 GB/s
        ==================================
        N=7808, REPEAT=128
        Size: 116 MB
        Time: 1044.46 us
        Throughput: 14942.68 GB/s
        ==================================
        N=7936, REPEAT=128
        Size: 120 MB
        Time: 832.36 us
        Throughput: 19370.10 GB/s
        ==================================
        N=8064, REPEAT=128
        Size: 124 MB
        Time: 1111.41 us
        Throughput: 14978.43 GB/s
        ==================================
        N=8192, REPEAT=128
        Size: 128 MB
        Time: 1172.95 us
        Throughput: 14646.76 GB/s
        ==================================
        N=16384, REPEAT=128
        Size: 512 MB
        Time: 7260.94 us
        Throughput: 9464.27 GB/s
*/

#include <random>
#include <vector>

#include "kittens.cuh"

using namespace kittens;

template <int _N, int _REPEAT, int _TILE_N=128>
struct globals {
    static constexpr int N = _N;
    static constexpr int REPEAT = _REPEAT;
    static constexpr int TILE_N = _TILE_N;

    using tile = st_bf<TILE_N, TILE_N>;
    using a_gl = gl<bf16, 1, 1, N, N, tile>;
    using b_gl = gl<bf16, 1, 1, N, N, tile>;

    a_gl a;
    b_gl b;

    __host__ dim3 grid() { return dim3(148); }
    __host__ dim3 block() { return dim3(1); }
    __host__ __device__ constexpr int dynamic_shared_memory() const { 
        return MAX_SHARED_MEMORY - 1024; 
    }
};

template <typename globals_t>
__global__ void kernel(const __grid_constant__ globals_t g) {
    constexpr int PIPE_DEPTH = g.dynamic_shared_memory() / sizeof(globals_t::tile);

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    auto (&tile)[PIPE_DEPTH] = al.allocate<typename globals_t::tile, PIPE_DEPTH>();

    __shared__ semaphore inputs_arrived[PIPE_DEPTH];
    #pragma unroll
    for (int i = 0; i < PIPE_DEPTH; i++)
        init_semaphore(inputs_arrived[i], 0, 1);

    constexpr int rblks = globals_t::N / globals_t::TILE_N;
    constexpr int cblks = globals_t::N / globals_t::TILE_N;
    constexpr int nblks = rblks * cblks;

    for (int i = 0; true; i++) {
        int task_idx = blockIdx.x + gridDim.x * i;
        if (task_idx >= nblks * globals_t::REPEAT) break;
        task_idx %= nblks;

        const int row = task_idx / cblks;
        const int col = task_idx % cblks;
        const int stage = i % PIPE_DEPTH;
        const int phasebit = ((i + PIPE_DEPTH) / PIPE_DEPTH) % 2;

        wait(inputs_arrived[stage], phasebit);
        tma::load_async(tile[stage], g.a, {row, col}, inputs_arrived[stage]);
        tma::expect_bytes(inputs_arrived[stage], sizeof(globals_t::tile));
    }
}

template <typename globals_t>
void run() {
    constexpr int N = globals_t::N;
    constexpr int REPEAT = globals_t::REPEAT;
    constexpr int SIZE = N * N * sizeof(bf16);

    printf("==================================\n");
    printf("N=%d, REPEAT=%d\n", N, REPEAT);
    printf("Size: %.0f MB\n", SIZE / 1024. / 1024.);

    // Sleep for 50 ms to limit power consumption and thermals
    usleep(50000);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<bf16> h_A(N * N);
    for (int i = 0; i < N * N; ++i) h_A[i] = __float2bfloat16(dis(gen));

    bf16 *d_A, *d_B;
    CUDACHECK(cudaMalloc(&d_A, SIZE));
    CUDACHECK(cudaMalloc(&d_B, SIZE));
    CUDACHECK(cudaMemcpy(d_A, h_A.data(), SIZE, cudaMemcpyHostToDevice));

    typename globals_t::a_gl a_gl{d_A, nullptr, nullptr, nullptr, nullptr};
    typename globals_t::b_gl b_gl{d_B, nullptr, nullptr, nullptr, nullptr};
    globals_t g{a_gl, b_gl};

    constexpr int num_warmups = 500;
    constexpr int num_iters = 100;
    cudaFuncSetAttribute(kernel<globals_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());

    for(int i = 0; i < num_warmups; i++)
        kernel<globals_t><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    for(int i = 0; i < num_iters; i++)
        kernel<globals_t><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avg_time = double(milliseconds) / 1000.0 / num_iters;
    double bps = double(REPEAT) * SIZE / avg_time;
    printf("Time: %.2f us\n", avg_time * 1e6);
    printf("Throughput: %.2f GB/s\n", bps / 1e9);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int device_id;
    CUDACHECK(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, device_id));
    printf("L2 Size=%.2f MB\n", (double)prop.l2CacheSize / 1024 / 1024);

    run<globals<2048, 128, 128>>();
    run<globals<3072, 128, 128>>();
    run<globals<4096, 128, 128>>();
    run<globals<4224, 128, 128>>();
    run<globals<4352, 128, 128>>();
    run<globals<4480, 128, 128>>();
    run<globals<4608, 128, 128>>();
    run<globals<4736, 128, 128>>();
    run<globals<4864, 128, 128>>();
    run<globals<4992, 128, 128>>();
    run<globals<5120, 128, 128>>();
    run<globals<5248, 128, 128>>();
    run<globals<5376, 128, 128>>();
    run<globals<5504, 128, 128>>();
    run<globals<5632, 128, 128>>();
    run<globals<5760, 128, 128>>();
    run<globals<5888, 128, 128>>();
    run<globals<6016, 128, 128>>();
    run<globals<6144, 128, 128>>();
    run<globals<6272, 128, 128>>();
    run<globals<6400, 128, 128>>();
    run<globals<6528, 128, 128>>();
    run<globals<6656, 128, 128>>();
    run<globals<6784, 128, 128>>();
    run<globals<6912, 128, 128>>();
    run<globals<7040, 128, 128>>();
    run<globals<7168, 128, 128>>();
    run<globals<7296, 128, 128>>();
    run<globals<7424, 128, 128>>();
    run<globals<7552, 128, 128>>();
    run<globals<7680, 128, 128>>();
    run<globals<7808, 128, 128>>();
    run<globals<7936, 128, 128>>();
    run<globals<8064, 128, 128>>();
    run<globals<8192, 128, 128>>();
    run<globals<16384, 128, 128>>();
}
