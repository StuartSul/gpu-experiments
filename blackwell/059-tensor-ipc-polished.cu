#include "gpu-experiments.cuh"

__global__ void set_value(float *ptr, float val, int N) {
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        ptr[i] = val;
}

void tensor_ipc_example_func(pybind11::object tensor, TensorIPC &tensor_ipc) {
    // Simple checks
    if (!pybind11::hasattr(tensor, "__class__"))
        throw std::runtime_error("Not a Python object.");
    if (tensor.attr("__class__").attr("__name__").cast<std::string>() != "Tensor")
        throw std::runtime_error("Not a torch.Tensor object.");
    if (!tensor.attr("is_contiguous")().cast<bool>())
        throw std::runtime_error("Tensor must be contiguous");
    if (tensor.attr("device").attr("type").cast<std::string>() != "cuda")
        throw std::runtime_error("Tensor must be on CUDA device");

    // Retrieve device pointer
    void *export_ptr = reinterpret_cast<void *>(tensor.attr("data_ptr")().cast<uint64_t>());

    // Export IPC handle
    std::vector<TensorIPCEntry> out_entries(tensor_ipc.local_world_size_);
    tensor_ipc.all_gather_ptrs(out_entries.data(), export_ptr);

    // Run kernel
    set_value<<<1, 1>>>(
        (float *)out_entries[(tensor_ipc.local_rank_ + 1) % tensor_ipc.local_world_size_].raw_ptr,
        (float)tensor_ipc.local_rank_,
        128 * 128
    );
    CUDACHECK(cudaDeviceSynchronize());
}

PYBIND11_MODULE(_C, m){
    pybind11::class_<TensorIPC>(m, "TensorIPC")
        .def(pybind11::init<int, int>());
    m.def("tensor_ipc_example_func", &tensor_ipc_example_func);
}
