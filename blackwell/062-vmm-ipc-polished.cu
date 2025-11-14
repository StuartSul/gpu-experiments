#include "kittens.cuh"
#include "pyutils/torchutils.cuh"

__global__ void kernel(float *ptr, int N) {
    for (int i = 0; i < N; i++) ptr[i] = 3.14;
}

void vmm_ipc(int local_rank, int local_world_size) {
    kittens::detail::ipc::enable_all_peer_access(local_world_size);
    kittens::KittensBroker broker(local_rank, local_world_size);

    // Main
    int N = 1024;
    void *raw_ptr;
    size_t size = N * N * sizeof(float);
    size_t allocated_size;
    kittens::detail::vmm::vm_alloc_map_set_access(&raw_ptr, &allocated_size, size, local_rank, local_world_size);
    using ipc_handle_t = kittens::detail::ipc::handle<kittens::detail::ipc::flavor::VMM>;
    ipc_handle_t ipc_handle; // file descriptor
    kittens::detail::ipc::export_handle(&ipc_handle, raw_ptr);

    std::vector<int> all_fds(local_world_size);
    broker.exchange_fds(all_fds.data(), ipc_handle.handle_);

    std::vector<void *> raw_ptrs;
    for (int i = 0; i < local_world_size; i++) {
        if (i == local_rank) {
            raw_ptrs.push_back(raw_ptr);
        } else {
            void *other_raw_ptr;
            kittens::detail::ipc::import_handle(&other_raw_ptr, *reinterpret_cast<ipc_handle_t*>(&all_fds[i]), allocated_size, local_world_size);
            raw_ptrs.push_back(other_raw_ptr);
        }
    }

    if (local_rank == 0) {
        CUDACHECK(cudaSetDevice(0));
        for (int i = 0; i < local_world_size; i++)
            kernel<<<1, 1>>>(reinterpret_cast<float *>(raw_ptrs[i]), N * N);
        CUDACHECK(cudaDeviceSynchronize());
    }
    broker.sync();

    // Load data from device to host and print first 10 elements
    float *host_data = new float[N * N];
    CUDACHECK(cudaSetDevice(local_rank));
    CUDACHECK(cudaMemcpy(host_data, raw_ptr, size, cudaMemcpyDeviceToHost)); 
    for (int i = 0; i < local_world_size; i++) {
        if (i == local_rank) {
            std::cout << local_rank << ": ";
            for (int j = 0; j < 10 && j < N * N; j++) {
                std::cout << host_data[j] << " ";
            }
            std::cout << std::endl;
        }
        broker.sync();
    }

    // Cleanup
    delete[] host_data;
    for (int i = 0; i < local_world_size; i++)
        kittens::detail::vmm::vm_unmap(raw_ptrs[i], allocated_size);
}

// To test with torchrun
PYBIND11_MODULE(_C, m){
    m.def("vmm_ipc", &vmm_ipc);
}
