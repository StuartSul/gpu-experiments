/*
    Creating multicast-disabled TKPT takes roughly 330 ms
    Creating multicast-enabled TKPT takes roughly 450 ms
*/

#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

__global__ void kernel(float *ptr, int N) {
    for (int i = 0; i < N; i++) ptr[i] = 3.14;
}

void vmm_ipc(int local_rank, int local_world_size) {
    int N = 1024;
    auto t2 = std::chrono::high_resolution_clock::now();
    kittens::py::TKParallelTensor t({N, N}, at::ScalarType::Float, local_rank, local_world_size, true);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Time to create parallel tensor: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "ms" << std::endl;

    if (local_rank == 1) {
        CUDACHECK(cudaSetDevice(0));
        // for (int i = 0; i < local_world_size; i++)
        //     kernel<<<1, 1>>>((float *)t.raw_ptrs_[i], N * N);
        kernel<<<1, 1>>>((float *)t.multicast_ptr_, N * N); // multicast
        CUDACHECK(cudaDeviceSynchronize());
    }
    t.brokers_.at({local_rank, local_world_size}).sync();

    // Load data from device to host and print first 10 elements
    float *host_data = new float[N * N];
    CUDACHECK(cudaSetDevice(local_rank));
    CUDACHECK(cudaMemcpy(host_data, t.raw_ptrs_[t.local_rank_], N * N * sizeof(float), cudaMemcpyDeviceToHost)); 
    for (int i = 0; i < local_world_size; i++) {
        if (i == local_rank) {
            std::cout << local_rank << ": ";
            for (int j = 0; j < 10 && j < N * N; j++) {
                std::cout << host_data[j] << " ";
            }
            std::cout << std::endl;
        }
        t.brokers_.at({local_rank, local_world_size}).sync();
    }

    // Cleanup
    delete[] host_data;
}

// To test with torchrun
PYBIND11_MODULE(_C, m){
    m.def("vmm_ipc", &vmm_ipc);
}
