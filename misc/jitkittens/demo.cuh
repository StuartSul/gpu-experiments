__global__ void add(const float *a, const float *b, float *c, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) c[index] = a[index] + b[index];
}
