#include "gpu-experiments.cuh"

__global__ void kernel(int *tensor) {
    printf("Block: %d/%d | Thread: %d/%d\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
    for (int i = 0; i < 128; i++) tensor[i] = i;
}

template <typename T>
__host__ static inline T *get_data_ptr(py::object tensor) {
    // Assumes the following about `tensor`
    // - is a torch.Tensor object
    // - is contiguous
    // - is on device
    // - has the correct shape
    return reinterpret_cast<T *>(tensor.attr("data_ptr")().cast<uintptr_t>());
}

__host__ static inline void launch_kernel(py::object tensor) {
    kernel<<<1, 1>>>(get_data_ptr<int>(tensor));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel", &launch_kernel);
}
