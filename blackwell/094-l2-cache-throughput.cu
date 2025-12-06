/*
    B200 Results:
        L2 Size=126.50 MB
          8 MB -> 20774.81 GB/s
         18 MB -> 20076.07 GB/s
         32 MB -> 20404.87 GB/s
         34 MB -> 20458.84 GB/s
         36 MB -> 20480.86 GB/s
         38 MB -> 20536.13 GB/s
         40 MB -> 20422.36 GB/s
         43 MB -> 20553.50 GB/s
         45 MB -> 20532.41 GB/s
         48 MB -> 20521.45 GB/s
         50 MB -> 20484.06 GB/s
         53 MB -> 20485.90 GB/s
         55 MB -> 20508.27 GB/s
         58 MB -> 20588.39 GB/s
         60 MB -> 20550.01 GB/s
         63 MB -> 20336.78 GB/s
         66 MB -> 20337.30 GB/s
         69 MB -> 20355.34 GB/s
         72 MB -> 20341.86 GB/s
         75 MB -> 20374.96 GB/s
         78 MB -> 20173.30 GB/s
         81 MB -> 19396.67 GB/s
         84 MB -> 17559.75 GB/s
         88 MB -> 14375.89 GB/s
         91 MB -> 12014.71 GB/s
         95 MB -> 11474.61 GB/s
         98 MB ->  9550.01 GB/s
        102 MB ->  8487.95 GB/s
        105 MB ->  7954.35 GB/s
        109 MB ->  7779.18 GB/s
        112 MB ->  7643.32 GB/s
        116 MB ->  7567.09 GB/s
        120 MB ->  7489.14 GB/s
        124 MB ->  7434.99 GB/s
        128 MB ->  7415.28 GB/s
        512 MB ->  7532.46 GB/s

    H100 Results:
        L2 Size=50.00 MB
          8 MB -> 8295.00 GB/s
         18 MB -> 8232.88 GB/s
         32 MB -> 8212.35 GB/s
         34 MB -> 8180.07 GB/s
         36 MB -> 6979.29 GB/s
         38 MB -> 5801.35 GB/s
         40 MB -> 4665.71 GB/s
         43 MB -> 4073.90 GB/s
         45 MB -> 3662.96 GB/s
         48 MB -> 3448.52 GB/s
         50 MB -> 3348.83 GB/s
         53 MB -> 3297.16 GB/s
         55 MB -> 3270.31 GB/s
         58 MB -> 3253.01 GB/s
         60 MB -> 3246.14 GB/s
         63 MB -> 3243.81 GB/s
         66 MB -> 3241.66 GB/s
         69 MB -> 3240.35 GB/s
         72 MB -> 3240.33 GB/s
         75 MB -> 3240.75 GB/s
         78 MB -> 3240.51 GB/s
         81 MB -> 3238.68 GB/s
         84 MB -> 3241.12 GB/s
         88 MB -> 3240.66 GB/s
         91 MB -> 3240.96 GB/s
         95 MB -> 3241.19 GB/s
         98 MB -> 3241.10 GB/s
        102 MB -> 3241.36 GB/s
        105 MB -> 3241.66 GB/s
        109 MB -> 3242.68 GB/s
        112 MB -> 3241.03 GB/s
        116 MB -> 3242.75 GB/s
        120 MB -> 3241.30 GB/s
        124 MB -> 3241.56 GB/s
        128 MB -> 3242.66 GB/s
        512 MB -> 3245.77 GB/s
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
