#include <random>
#include <vector>

#include "kittens.cuh"

using namespace kittens;

constexpr int DEVICE_ID = 0;
constexpr int N = 6144; // for roughly 1/2 the L2 size
constexpr int TILE_N = 128;
constexpr int REPEAT = 64;

struct globals {
    using tile = st_bf<TILE_N, TILE_N>;
    using a_gl = gl<bf16, 1, 1, N, N, tile>;
    using b_gl = gl<bf16, 1, 1, N, N, tile>;

    a_gl a;
    b_gl b;

    __host__ dim3 grid() { return dim3(N / TILE_N, N / TILE_N, REPEAT); }
    __host__ dim3 block() { return dim3(1); }
    __host__ int dynamic_shared_memory() { return TILE_N * TILE_N * sizeof(bf16) + 1024; }
};

__global__ void kernel(const __grid_constant__ globals g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    globals::tile &tile = al.allocate<globals::tile>();

    __shared__ semaphore inputs_arrived;
    init_semaphore(inputs_arrived, 0, 1);

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    // Just do load for checking L2 effect
    tma::load_async(tile, g.a, {row, col}, inputs_arrived);
    // tma::expect_bytes(inputs_arrived, sizeof(tile));
    // wait(inputs_arrived, 0);

    // tma::store_async(g.b, tile, {row, col});
}

int main() {
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, DEVICE_ID));
    printf("L2 Size: %.2f MB\n", (double)prop.l2CacheSize / 1024 / 1024);
    printf("Matrix Size: %.2f MB\n", (double)N * N * sizeof(bf16) / 1024 / 1024);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<bf16> h_A(N * N);
    for (int i = 0; i < N * N; ++i) h_A[i] = __float2bfloat16(dis(gen));

    bf16 *d_A, *d_B;
    CUDACHECK(cudaMalloc(&d_A, N * N * sizeof(bf16)));
    CUDACHECK(cudaMalloc(&d_B, N * N * sizeof(bf16)));
    CUDACHECK(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(bf16), cudaMemcpyHostToDevice));

    globals::a_gl a_gl{d_A, nullptr, nullptr, nullptr, nullptr};
    globals::b_gl b_gl{d_B, nullptr, nullptr, nullptr, nullptr};
    globals g{a_gl, b_gl};

    constexpr int num_warmups = 500;
    constexpr int num_iters = 100;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());

    for(int i = 0; i < num_warmups; i++)
        kernel<<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    for(int i = 0; i < num_iters; i++)
        kernel<<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avg_time = double(milliseconds) / 1000.0 / num_iters;
    double bps = double(REPEAT) * N * N * sizeof(bf16) / avg_time;
    std::cout << (avg_time * 1e6) << " us, " << (bps / 1e9) << " GB/s\n";

    std::vector<bf16> h_B(N * N);
    CUDACHECK(cudaMemcpy(h_B.data(), d_B, N * N * sizeof(bf16), cudaMemcpyDeviceToHost));
    int error_count = 0;
    for (int i = 0; i < N * N; ++i) {
        if (h_B[i] != h_A[i]) error_count++;
    }
    std::cout << "Error count: " << error_count << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
