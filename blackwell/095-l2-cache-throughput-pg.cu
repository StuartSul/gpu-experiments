/*
    B200 Results:
        L2 Size=126.50 MB
          8 MB -> 18698.17 GB/s
         18 MB -> 19025.33 GB/s
         32 MB -> 19272.33 GB/s
         34 MB -> 19134.60 GB/s
         36 MB -> 19163.01 GB/s
         38 MB -> 19322.96 GB/s
         40 MB -> 19261.48 GB/s
         43 MB -> 18981.67 GB/s
         45 MB -> 19324.36 GB/s
         48 MB -> 19310.86 GB/s
         50 MB -> 18823.83 GB/s
         53 MB -> 19221.23 GB/s
         55 MB -> 19410.41 GB/s
         58 MB -> 19182.72 GB/s
         60 MB -> 19348.00 GB/s
         63 MB -> 19328.72 GB/s
         66 MB -> 19246.84 GB/s
         69 MB -> 19503.92 GB/s
         72 MB -> 19267.08 GB/s
         75 MB -> 19173.64 GB/s
         78 MB -> 18866.75 GB/s
         81 MB -> 18695.86 GB/s
         84 MB -> 17787.46 GB/s
         88 MB -> 19469.45 GB/s
         91 MB -> 17537.25 GB/s
         95 MB -> 17177.20 GB/s
         98 MB -> 16683.20 GB/s
        102 MB -> 19671.48 GB/s
        105 MB -> 15693.39 GB/s
        109 MB -> 16918.17 GB/s
        112 MB -> 15427.73 GB/s
        116 MB -> 14942.68 GB/s
        120 MB -> 19370.10 GB/s
        124 MB -> 14978.43 GB/s
        128 MB -> 14646.76 GB/s
        512 MB ->  9464.27 GB/s

    H100 Results:
        L2 Size=50.00 MB
          8 MB -> 7630.41 GB/s
         18 MB -> 7665.68 GB/s
         32 MB -> 7707.10 GB/s
         34 MB -> 7549.78 GB/s
         36 MB -> 7537.34 GB/s
         38 MB -> 7239.34 GB/s
         40 MB -> 6553.62 GB/s
         43 MB -> 6026.10 GB/s
         45 MB -> 7123.70 GB/s
         48 MB -> 6336.65 GB/s
         50 MB -> 6218.25 GB/s
         53 MB -> 5672.52 GB/s
         55 MB -> 5245.19 GB/s
         58 MB -> 7262.34 GB/s
         60 MB -> 4670.33 GB/s
         63 MB -> 5260.36 GB/s
         66 MB -> 5053.92 GB/s
         69 MB -> 4816.37 GB/s
         72 MB -> 4597.88 GB/s
         75 MB -> 4518.04 GB/s
         78 MB -> 4950.71 GB/s
         81 MB -> 4585.84 GB/s
         84 MB -> 4411.55 GB/s
         88 MB -> 4224.30 GB/s
         91 MB -> 4041.55 GB/s
         95 MB -> 4291.53 GB/s
         98 MB -> 3997.72 GB/s
        102 MB -> 4328.63 GB/s
        105 MB -> 3884.11 GB/s
        109 MB -> 3808.27 GB/s
        112 MB -> 3904.09 GB/s
        116 MB -> 3612.59 GB/s
        120 MB -> 3919.75 GB/s
        124 MB -> 3566.53 GB/s
        128 MB -> 1771.63 GB/s
        512 MB -> 3231.32 GB/s
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
