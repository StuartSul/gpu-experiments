/*
    WIP. Does not work yet
*/

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>

#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Failed: CUDA error %s:%d '%s'\n",    \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// perform Allreduce using ring
__global__ void ring_reduce(
    int *dst,
    const int *src,
    size_t N,
    uint64_t *signal
) {
    int rank = nvshmem_my_pe();
    int world_size = nvshmem_n_pes();
    int peer_rank = (rank + 1) % world_size;

    size_t N_per_dev = N / world_size;
    size_t N_per_block = N_per_dev / gridDim.x;
    size_t base_idx = rank * N_per_dev + blockIdx.x * N_per_block;

    if (base_idx > N)
        return;

    src = src + base_idx;
    dst = dst + base_idx;
    signal = signal + blockIdx.x;

    // Reduce phase
    for (int step = 0; step < world_size - 1; step++) {
        if (rank != 0) {
            if (threadIdx.x == 0)
                nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, step + 1);
            __syncthreads();

            for (size_t i = threadIdx.x; i < N_per_dev; i += blockDim.x) {
                dst[i] = dst[i] + src[i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
            nvshmem_int_put_signal_nbi(dst, (rank == 0) ? src : dst, N_per_dev, signal, 1, NVSHMEM_SIGNAL_ADD, peer_rank);

        src = src + N_per_dev;
        dst = dst + N_per_dev;
    }

    // Broadcast phase
    dst = dst - num_chunks * chunk_elems;
    if (threadIdx.x == 0) {
        for (size_t chunk = 0; chunk < num_chunks; chunk++) {
            if (rank < world_size - 1)
                nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, (rank == 0) ? chunk + 1 : num_chunks + chunk + 1);
            if (rank < world_size - 2)
                nvshmem_int_put_signal_nbi(dst, dst, chunk_elems, signal, 1, NVSHMEM_SIGNAL_ADD, peer);
            dst = dst + chunk_elems;
        }
        *signal = 0;
    }
}

static size_t SIZE = 1024 * 1024 * 32;
static constexpr int NUM_BLOCKS = 32;
static constexpr int NUM_THREADS = 512;
static constexpr int NUM_WARMUPS = 1;
static constexpr int NUM_ITERS = 4;

int main(int argc, char **argv) {
    // Initialize MPI
    int rank;
    int world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Initialize NVSHMEM
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Retrieve NVSHMEM PE info
    int current_pe = nvshmem_my_pe();
    int num_pes = nvshmem_n_pes();
    int current_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    assert(current_pe_node == current_pe);
    assert(current_pe == rank);
    assert(num_pes == world_size);

    // Set CUDA device
    CUDACHECK(cudaSetDevice(current_pe_node));

    // Create CUDA stream and events
    cudaStream_t stream;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    CUDACHECK(cudaEventCreate(&start_event));
    CUDACHECK(cudaEventCreate(&stop_event));
    CUDACHECK(cudaStreamCreate(&stream));

    // Allocate and initialize host memory
    size_t N = SIZE / sizeof(int);
    int *host_buffer = reinterpret_cast<int *>(malloc(SIZE));
    for (size_t i = 0; i < N; i++)
        host_buffer[i] = i;

    // Allocate and initialize device memory
    int *dst = reinterpret_cast<int *>(nvshmem_malloc(SIZE));
    int *src = reinterpret_cast<int *>(nvshmem_malloc(SIZE));
    uint64_t *signal = (uint64_t *)nvshmem_calloc(NUM_BLOCKS, sizeof(uint64_t));
    CUDACHECK(cudaMemcpyAsync(src, host_buffer, SIZE, cudaMemcpyHostToDevice, stream));
    nvshmemx_barrier_all_on_stream(stream);

    // Kernel configuration
    dim3 gridDim(NUM_BLOCKS);
    dim3 blockDim(NUM_THREADS);
    void *args[] = {&dst, &src, &N, &signal};

    // Check correctness
    nvshmemx_collective_launch((const void *)ring_reduce, gridDim, blockDim, args, 0, stream);
    nvshmemx_barrier_all_on_stream(stream);
    CUDACHECK(cudaMemcpyAsync(host_buffer, dst, SIZE, cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < N; i++) {
        if (host_buffer[i] != i * world_size)
            printf("Error on rank %d: data[%d] = %d (expected %d)\n", rank, i, host_buffer[i], i * world_size);
    }

    // Warmups
    for (int i = 0; i < NUM_WARMUPS; i++) {
        nvshmemx_collective_launch((const void *)ring_reduce, gridDim, blockDim, args, 0, stream);
        nvshmemx_barrier_all_on_stream(stream);
    }
    CUDACHECK(cudaStreamSynchronize(stream));

    // Benchmark
    CUDACHECK(cudaEventRecord(start_event, stream));
    for (int i = 0; i < NUM_ITERS; i++) {
        nvshmemx_collective_launch((const void *)ring_reduce, gridDim, blockDim, args, 0, stream);
        nvshmemx_barrier_all_on_stream(stream);
    }
    CUDACHECK(cudaEventRecord(stop_event, stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    // Print result
    if (rank == 0) {
        float total_ms;
        CUDACHECK(cudaEventElapsedTime(&total_ms, start_event, stop_event));
        printf("%zuB \t %fms\n", SIZE, total_ms / NUM_ITERS);
    }

    // Clean up
    CUDACHECK(cudaEventDestroy(start_event));
    CUDACHECK(cudaEventDestroy(stop_event));
    nvshmem_free(dst);
    nvshmem_free(src);
    nvshmem_free(signal);
    free(host_buffer);

    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
