/*
    B200 Results:
        L2 Size=126.50 MB
        ==================================
        N=2048, REPEAT=128
        Size: 8 MB
        Time: 51.78 us
        Throughput: 20736.68 GB/s
        ==================================
        N=3072, REPEAT=128
        Size: 18 MB
        Time: 120.49 us
        Throughput: 20050.75 GB/s
        ==================================
        N=4096, REPEAT=128
        Size: 32 MB
        Time: 210.54 us
        Throughput: 20400.25 GB/s
        ==================================
        N=5120, REPEAT=128
        Size: 50 MB
        Time: 327.61 us
        Throughput: 20484.26 GB/s
        ==================================
        N=6144, REPEAT=128
        Size: 72 MB
        Time: 475.15 us
        Throughput: 20338.35 GB/s
        ==================================
        N=6272, REPEAT=128
        Size: 75 MB
        Time: 494.23 us
        Throughput: 20376.13 GB/s
        ==================================
        N=6400, REPEAT=128
        Size: 78 MB
        Time: 518.99 us
        Throughput: 20204.16 GB/s
        ==================================
        N=6528, REPEAT=128
        Size: 81 MB
        Time: 560.03 us
        Throughput: 19480.11 GB/s
        ==================================
        N=6656, REPEAT=128
        Size: 84 MB
        Time: 645.42 us
        Throughput: 17572.17 GB/s
        ==================================
        N=6784, REPEAT=128
        Size: 88 MB
        Time: 813.70 us
        Throughput: 14479.24 GB/s
        ==================================
        N=6912, REPEAT=128
        Size: 91 MB
        Time: 1010.32 us
        Throughput: 12105.72 GB/s
        ==================================
        N=7040, REPEAT=128
        Size: 95 MB
        Time: 1138.33 us
        Throughput: 11145.98 GB/s
        ==================================
        N=7168, REPEAT=128
        Size: 98 MB
        Time: 1377.47 us
        Throughput: 9548.91 GB/s
        ==================================
        N=7296, REPEAT=128
        Size: 102 MB
        Time: 1605.14 us
        Throughput: 8489.76 GB/s
        ==================================
        N=7424, REPEAT=128
        Size: 105 MB
        Time: 1773.85 us
        Throughput: 7954.24 GB/s
        ==================================
        N=7552, REPEAT=128
        Size: 109 MB
        Time: 1876.99 us
        Throughput: 7778.61 GB/s
        ==================================
        N=7680, REPEAT=128
        Size: 112 MB
        Time: 1975.55 us
        Throughput: 7643.20 GB/s
        ==================================
        N=7808, REPEAT=128
        Size: 116 MB
        Time: 2062.12 us
        Throughput: 7568.44 GB/s
        ==================================
        N=7936, REPEAT=128
        Size: 120 MB
        Time: 2152.71 us
        Throughput: 7489.60 GB/s
        ==================================
        N=8064, REPEAT=128
        Size: 124 MB
        Time: 2239.37 us
        Throughput: 7433.89 GB/s
        ==================================
        N=8192, REPEAT=128
        Size: 128 MB
        Time: 2316.96 us
        Throughput: 7414.82 GB/s
        ==================================
        N=16384, REPEAT=128
        Size: 512 MB
        Time: 9122.68 us
        Throughput: 7532.81 GB/s
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

    __host__ dim3 grid() { return dim3(N / TILE_N, N / TILE_N, REPEAT); }
    __host__ dim3 block() { return dim3(1); }
    __host__ int dynamic_shared_memory() { return TILE_N * TILE_N * sizeof(bf16) + 1024; }
};

template <typename globals_t>
__global__ void kernel(const __grid_constant__ globals_t g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    auto &tile = al.allocate<typename globals_t::tile>();

    __shared__ semaphore inputs_arrived;
    init_semaphore(inputs_arrived, 0, 1);

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // No need to wait as kernel will flush before exit
    tma::load_async(tile, g.a, {row, col}, inputs_arrived);
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
