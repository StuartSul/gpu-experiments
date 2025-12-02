/*
Results (ablation is accumulative)

Original gemm
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 18.4918 us
Achieved performance: 116.131 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 28.7286 us
Achieved performance: 598.005 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 104.502 us
Achieved performance: 1315.18 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 740.274 us
Achieved performance: 1485.28 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 6009.25 us
Achieved performance: 1463.76 TFLOPs

No consumer
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 14.3853 us
Achieved performance: 149.283 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 24.616 us
Achieved performance: 697.915 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 89.056 us
Achieved performance: 1543.29 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 711.706 us
Achieved performance: 1544.9 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 5952.93 us
Achieved performance: 1477.61 TFLOPs

No consumer warpgroup
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 14.3834 us
Achieved performance: 149.303 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 24.5453 us
Achieved performance: 699.926 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 84.0714 us
Achieved performance: 1634.79 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 716.1 us
Achieved performance: 1535.42 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 5969.63 us
Achieved performance: 1473.47 TFLOPs

Launcher does not wait for loads & no loads issued
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.5978 us
Achieved performance: 170.466 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 22.5664 us
Achieved performance: 761.303 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 75.7872 us
Achieved performance: 1813.49 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 503.112 us
Achieved performance: 2185.42 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 3984.58 us
Achieved performance: 2207.53 TFLOPs

Loader completely gone, no semaphores
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3568 us
Achieved performance: 173.79 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 21.5731 us
Achieved performance: 796.355 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 74.9674 us
Achieved performance: 1833.32 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 501.422 us
Achieved performance: 2192.79 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 3985.2 us
Achieved performance: 2207.19 TFLOPs

Use only half of TMEM (i.e. only one thread launches MMAs)
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.25024 us
Achieved performance: 260.293 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3504 us
Achieved performance: 1391.04 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 75.6429 us
Achieved performance: 1816.94 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 500.511 us
Achieved performance: 2196.78 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 3984.69 us
Achieved performance: 2207.47 TFLOPs

Increasing to 6 stages
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.25952 us
Achieved performance: 260.001 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3389 us
Achieved performance: 1392.34 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 75.8394 us
Achieved performance: 1812.24 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 500.551 us
Achieved performance: 2196.6 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 3985.43 us
Achieved performance: 2207.06 TFLOPs

Removing task iteration calculation
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.24416 us
Achieved performance: 260.485 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3334 us
Achieved performance: 1392.95 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 75.4883 us
Achieved performance: 1820.67 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 500.544 us
Achieved performance: 2196.63 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 3985.54 us
Achieved performance: 2207 TFLOPs

Remove everyone::sync()
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.26208 us
Achieved performance: 259.92 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3379 us
Achieved performance: 1392.44 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 74.344 us
Achieved performance: 1848.69 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 501.124 us
Achieved performance: 2194.09 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 3985.38 us
Achieved performance: 2207.09 TFLOPs

(Experiment) Increaseing to cluster size 4, but still just pure matmul
Learning: single-wave matmul is NOT affected, but multi-wave is. Apparently scheduling slows down with tb clusters
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.25152 us
Achieved performance: 260.253 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3488 us
Achieved performance: 1391.22 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 110.874 us
Achieved performance: 1239.6 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 964.371 us
Achieved performance: 1140.13 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 7806.02 us
Achieved performance: 1126.83 TFLOPs

Pure 1 CTA matmul
Learning: speedup of 2-CTA matmul is purely from reduced HBM/SMEM traffic
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.29888 us
Achieved performance: 258.768 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.328 us
Achieved performance: 1393.56 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 73.9597 us
Achieved performance: 1858.3 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 499.569 us
Achieved performance: 2200.92 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 3971.18 us
Achieved performance: 2214.98 TFLOPs

1CTA matmul & no cluster_dims
Learning: No gain here at all
./087-pure-mma-speed.out
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.2528 us
Achieved performance: 260.213 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3357 us
Achieved performance: 1392.7 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 73.8061 us
Achieved performance: 1862.16 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 499.702 us
Achieved performance: 2200.33 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 3990.02 us
Achieved performance: 2204.52 TFLOPs

Non-persistent 2D grid
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.25024 us
Achieved performance: 260.293 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3302 us
Achieved performance: 1393.31 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 78.0547 us
Achieved performance: 1760.8 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 516.254 us
Achieved performance: 2129.79 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 4036.43 us
Achieved performance: 2179.18 TFLOPs

Non-persistent 2D grid + 2D 4-cluster
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.2384 us
Achieved performance: 260.668 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3222 us
Achieved performance: 1394.22 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 78.0877 us
Achieved performance: 1760.06 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 588.203 us
Achieved performance: 1869.27 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 4538.76 us
Achieved performance: 1938 TFLOPs

Non-persistent 1D grid
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.24576 us
Achieved performance: 260.435 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.327 us
Achieved performance: 1393.67 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 77.8829 us
Achieved performance: 1764.69 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 516.193 us
Achieved performance: 2130.04 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 4036.45 us
Achieved performance: 2179.17 TFLOPs

Non-persistent 1D grid + 1D 4-cluster
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.25216 us
Achieved performance: 260.233 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3498 us
Achieved performance: 1391.11 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 78.0528 us
Achieved performance: 1760.85 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 588.163 us
Achieved performance: 1869.4 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 4538.77 us
Achieved performance: 1937.99 TFLOPs

1D CLC persistent grid with 2-cluster
Learning: not much benefit. Only like 1% improvement on larger shapes with multiple waves. But I assume things would change with HBM traffic
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.25568 us
Achieved performance: 260.122 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3482 us
Achieved performance: 1391.29 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 75.7888 us
Achieved performance: 1813.45 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 500.889 us
Achieved performance: 2195.12 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 3967.27 us
Achieved performance: 2217.17 TFLOPs

1D CLC persistent grid with 4-cluster
Learning: slightly slows down, but def better than plain persistent grid with 4-cluster! Much more usable! But why?
--------------------  M=1024 N=1024 K=1024  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 8.24512 us
Achieved performance: 260.455 TFLOPs
--------------------  M=2048 N=2048 K=2048  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 12.3354 us
Achieved performance: 1392.73 TFLOPs
--------------------  M=4096 N=4096 K=4096  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 75.8006 us
Achieved performance: 1813.16 TFLOPs
--------------------  M=8192 N=8192 K=8192  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 570.82 us
Achieved performance: 1926.2 TFLOPs
--------------------  M=16384 N=16384 K=16384  --------------------
Block size: 256x256x64
Allocated host memory
Initialized matrices
Allocated device memory
Copied matrices to device
Avg Kernel execution time: 4461.99 us
Achieved performance: 1971.34 TFLOPs
*/

#include "kittens.cuh"
#include <iostream>

using namespace kittens;

constexpr int NUM_THREADS = 1;

static constexpr int Mb = 128;
static constexpr int Nb = 256;
static constexpr int Kb = 64;

constexpr int PIPE_DEPTH = 6;

constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - 1024;

constexpr int CLUSTER_SIZE = 2;

struct matmul_globals {
    using a_tile = st_bf<Mb, Kb>;
    using b_tile = st_bf<Nb/2, Kb>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
    using d_gl = gl<bf16, 1, 1, -1, -1>;

    a_gl a;
    b_gl b;
    d_gl d;

    __host__ __inline__ dim3 grid() {
        return dim3(148);
    }
};

__global__ __cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(NUM_THREADS, 1)
void matmul(const __grid_constant__ matmul_globals g) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    constexpr int CLUSTER_M = CLUSTER_SIZE*Mb, CLUSTER_N = Nb;
    const int num_tasks = g.d.rows() / CLUSTER_M * g.d.cols() / CLUSTER_N;
    const int iters_per_task = g.a.cols() / Kb;

    const int cluster_id = clusterIdx().x;
    const int cta_rank = cluster_ctarank();

    using a_tile = matmul_globals::a_tile;
    using b_tile = matmul_globals::b_tile;
    
    a_tile (&a_smem)[PIPE_DEPTH] = al.allocate<a_tile, PIPE_DEPTH>();
    b_tile (&b_smem)[PIPE_DEPTH] = al.allocate<b_tile, PIPE_DEPTH>();

    tensor_allocator<1, 2> tm_alloc{};
    using d_tt_t = tt<float, Mb, Nb>;
    everyone::tma::cluster::sync();

    if(cta_rank%2 == 0) {
        d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);
        for(int task_iter=cluster_id; task_iter < num_tasks; task_iter+=gridDim.x/CLUSTER_SIZE) {
            int input_ring = 0;
            mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring]);
            input_ring=ring_advance<PIPE_DEPTH>(input_ring);
            for(int idx = 1; idx < iters_per_task; idx++) {
                mma2_ABt(d_tt, a_smem[input_ring], b_smem[input_ring]);
                input_ring=ring_advance<PIPE_DEPTH>(input_ring);
            }
        }
    }
}


#include <iostream>
#include <random>

constexpr int NCU = true;

void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K) {
    using globals  = matmul_globals;
    typename globals::a_gl Ag{d_A, nullptr, nullptr, M, K};
    typename globals::b_gl Bg{d_B, nullptr, nullptr, N, K};
    typename globals::d_gl Dg{d_C, nullptr, nullptr, M, N};
    globals G{Ag, Bg, Dg};
    matmul<<<G.grid(), NUM_THREADS, DYNAMIC_SHARED_MEMORY>>>(G);
}

int run_benchmark(size_t M, size_t N, size_t K) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Block size: " << Mb*2 << "x" << Nb << "x" << Kb << "\n";
    std::cout << "Num tasks: " << (M/Mb*N/Nb) << "\n";

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C;
    CUDACHECK(cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16)));
    CUDACHECK(cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16)));
    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device. Otherwise GPU does not truly perform a matmul
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    CUDACHECK(cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice));
    std::cout << "Copied matrices to device" << std::endl;

    // Set kernel attributes
    cudaFuncSetAttribute(matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, DYNAMIC_SHARED_MEMORY);

    // Warmup
    for(int i = 0; i < (NCU ? 0 : 500); i++)
        inner_run(d_A, d_B, d_C, M, N, K);

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaEventRecord(start));
    constexpr int ITERS = (NCU ? 1 : 100);
    for(int i = 0; i < ITERS; i++)
        inner_run(d_A, d_B, d_C, M, N, K);
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double useconds = milliseconds * 1000.0 / ITERS;
    double flops = double(2.0) * M * N * K; // 2 FLOPs per multiply-add
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main() {
    int N;
    if (NCU) {
        N = 4096;
        run_benchmark(N, N, N);
    } else {
        N = 1024;
        run_benchmark(N, N, N);
        N = 2048;
        run_benchmark(N, N, N);
        N = 4096;
        run_benchmark(N, N, N);
        N = 8192;
        run_benchmark(N, N, N);
        N = 16384;
        run_benchmark(N, N, N);
    }
    return 0;
}
