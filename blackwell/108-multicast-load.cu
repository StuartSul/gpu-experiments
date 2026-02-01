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

template <int _Mb, int _Nb, int _Kb, int _NUM_BLOCKS, int _CLUSTER_PREFERRED, int _CLUSTER_MINIMUM>
struct config {
    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;

    static constexpr int NUM_BLOCKS = _NUM_BLOCKS;
    static constexpr int NUM_THREADS = 128;

    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

    static constexpr int CLUSTER_PREFERRED = _CLUSTER_PREFERRED;
    static constexpr int CLUSTER_MINIMUM = _CLUSTER_MINIMUM;

    static constexpr int LOAD_PIPE_DEPTH = 4;
    static constexpr int SUPERGROUP_SIZE = 8;

};

template <typename C>
struct globals {
    using a_tile = st_fp8e4m3<C::Mb/2, C::Kb>;
    using b_tile = st_fp8e4m3<C::Nb/2, C::Kb>;

    using a_gl = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;

    a_gl a;
    b_gl b;
};

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void kernel(const __grid_constant__ globals<C> g) {
    using G = globals<C>;
    using A_tile = typename G::a_tile;
    using B_tile = typename G::b_tile;

    // Allocate shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator sm_allocator((int*)&__shm[0]);
    static_assert((sizeof(A_tile) + sizeof(B_tile)) * C::LOAD_PIPE_DEPTH + 1024 <= C::DYNAMIC_SHARED_MEMORY);
    A_tile (&A_tiles)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<A_tile, C::LOAD_PIPE_DEPTH>();
    B_tile (&B_tiles)[C::LOAD_PIPE_DEPTH] = sm_allocator.allocate<B_tile, C::LOAD_PIPE_DEPTH>();

    // Declare tensor memory
    tensor_allocator<1, 2, false> tm_allocator;

    // Set up mbarriers
    __shared__ uint32_t tmem_addr;
    __shared__ semaphore inputs_arrived[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore inputs_finished[C::LOAD_PIPE_DEPTH];
    __shared__ semaphore tmem_finished;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < C::LOAD_PIPE_DEPTH; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(tmem_finished, 0, 1);
    }
    everyone::tma::cluster::arrive_aligned();

    const int cta_id = cluster_ctarank();
    const int cta_parity = cta_id & 0b1;
    const int cluster_size = cluster_nctarank();
    const int cluster_id = clusterid().x*(cluster_size/2) + cta_id/2;
    const int rblks = g.a.rows() / C::Mb;
    const int cblks = g.b.rows() / C::Nb;
    const int num_blocks = cblks * rblks;
    const int num_iters_per_block = g.a.cols() / C::Kb;

    uint32_t stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    // Main divergence
    if (warpgroup::warpid() == 3 && warp::elect_leader()) {
        // Load input tiles to shared memory
        pdl::wait();
        everyone::tma::cluster::wait();
        for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / 2) {
            int2 tile_coord = get_swizzled_2d_idx<C::SUPERGROUP_SIZE>(rblks, cblks, block_idx);

            for (int i = 0; i < num_iters_per_block; ++i) {
                tma::cluster::wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                update_phasebit<1>(phasebits, stage);
                tma::cluster::load_async(A_tiles[stage], g.a, {tile_coord.x * 2 + cta_parity, i}, inputs_arrived[stage], (uint16_t)(1<<cta_id), cta_id&(~0b1));
                tma::cluster::load_async(B_tiles[stage], g.b, {tile_coord.y * 2 + cta_parity, i}, inputs_arrived[stage], (uint16_t)(1<<cta_id), cta_id&(~0b1));
                stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
            }
        }
    } else if (warpgroup::warpid() == 0) {
        // Launch tensor core matrix multiply
        everyone::tma::cluster::wait();
        tm_allocator.provision(tmem_addr);
        tm_allocator.set_addr(tmem_addr);
        auto out_tm = tm_allocator.template allocate<full_tt_fl<C::Nb>>(0);
        if (cta_parity == 0 && warp::elect_leader()) {
            for (int block_idx = cluster_id; block_idx < num_blocks; block_idx += gridDim.x / 2) {
                for (int i = 0; i < num_iters_per_block; i++) {
                    tma::cluster::expect_bytes(inputs_arrived[stage], 2*(sizeof(A_tile)+sizeof(B_tile)));
                    tma::cluster::wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    if (i == 0) mm2_ABt(out_tm, A_tiles[stage], B_tiles[stage]);
                    else       mma2_ABt(out_tm, A_tiles[stage], B_tiles[stage]);
                    tensor_commit<2>(inputs_finished[stage], 0b11<<cta_id);
                    stage = (stage + 1) % C::LOAD_PIPE_DEPTH;
                }
            }
            tensor_commit<2>(tmem_finished, 0b11<<cta_id);
        }
        tma::cluster::wait(tmem_finished, 0);
        tm_allocator.deprovision();
    }
}

template <typename C>
__host__ double run(size_t M, size_t N, size_t K) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "CLUSTER_PREFERRED=" << C::CLUSTER_PREFERRED << " CLUSTER_MINIMUM=" << C::CLUSTER_MINIMUM << std::endl;

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

    CUDACHECK(cudaFuncSetAttribute(kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::DYNAMIC_SHARED_MEMORY));
    CUDACHECK(cudaFuncSetAttribute(kernel<C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

    dim3 cluster_preferred(C::CLUSTER_PREFERRED, 1, 1);
    dim3 cluster_minimum(C::CLUSTER_MINIMUM, 1, 1);
    LaunchConfig<true, true> launch_config(C::NUM_BLOCKS, C::NUM_THREADS, C::DYNAMIC_SHARED_MEMORY, 0, 
                                           cluster_preferred, cluster_minimum);

    constexpr int num_warmups = 100;
    constexpr int num_iters = 500;

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

    run<config<256, 256, 128, 148, 2, 2>>(N, N, N);
    // run<config<256, 256, 128, 148, 4, 2>>(N, N, N);

    return 0;
}
