#include "kittens.cuh"

using namespace kittens;

template <typename T>
__global__ void fill_kernel(T* data, size_t count, uint64_t seed, float min_val, float max_val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint64_t x = seed + idx;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        float u = (float)(x >> 40) * (1.0f / 16777216.0f);
        float val = u * (max_val - min_val) + min_val;
        data[idx] = base_types::convertor<T, float>::convert(val);
    }
}

template <int _Mb, int _Nb, int _Kb>
struct config {
    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    static constexpr int NUM_THREADS = 128;
};

template <typename C>
struct globals {
    using a_tile = st_fp8e4m3<C::Mb, C::Kb>;
    using b_tile = st_fp8e4m3<C::Nb, C::Kb>;

    using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;

    a_gl a;
    b_gl b;

    __host__ __inline__ dim3 grid() { return dim3(148); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return MAX_SHARED_MEMORY - 1024; }
};

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals<C> g) {
    // to be implemented
}

template <typename C>
__host__ double run(size_t M, size_t N, size_t K) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";

    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = (size_t(M) * K + size_t(N) * K) * sizeof(__nv_fp8_e4m3);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    std::vector<__nv_fp8_e4m3*> d_A(arg_group_count);
    std::vector<__nv_fp8_e4m3*> d_B(arg_group_count);
    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M*K*sizeof(__nv_fp8_e4m3)));
        CUDACHECK(cudaMalloc(&d_B[i], K*N*sizeof(__nv_fp8_e4m3)));
    }
    std::cout << "Allocated device memory" << std::endl;

    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill_kernel<<<(M*K+255)/256, 256>>>(d_A[i], M*K, seed + i*100, -448.0f, 448.0f);
        fill_kernel<<<(K*N+255)/256, 256>>>(d_B[i], K*N, seed + i*100 + 1, -448.0f, 448.0f);
    }
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialized matrices on device" << std::endl;

    std::vector<globals<C>> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename globals<C>::a_gl Ag{d_A[i], nullptr, nullptr, M, K};
        typename globals<C>::b_gl Bg{d_B[i], nullptr, nullptr, N, K};
        g.push_back(globals<C>{Ag, Bg});
    }

    CUDACHECK(cudaFuncSetAttribute(kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g[0].dynamic_shared_memory()));

    LaunchConfig<true, true> launch_config(g[0].grid(), g[0].block(), g[0].dynamic_shared_memory(), 0, 2);

    constexpr int num_warmups = 500;
    constexpr int num_iters = 100;

    for(int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel<C>, g[idx]);
    }

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for(int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        cudaLaunchKernelEx(launch_config, kernel<C>, g[idx]);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

__host__ int main() {
    int N = 16384;
    run<config<256, 256, 128>>(N, N, N);
    return 0;
}
