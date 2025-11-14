import numpy as np
import torch
torch.manual_seed(42)
torch.set_printoptions(sci_mode=False)

# Import our Python bindings
from _C import bf16_matmul


# Matrix dimensions
M = 64
K = 16384
N = 16384

# Generate random matrices
print("Generating inputs...")
A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

# Run kernel
print("Launching kernel...")
bf16_matmul(A, B, C)
torch.cuda.synchronize()

# Check correctness
C_ref = A @ B.T
abs_diff = torch.abs(C_ref - C)
print(f"Max adiff  : {abs_diff.max()}")
print(f"Mean adiff : {abs_diff.mean()}")
print(f"Mean       : {C.abs().mean()}")
print(f"Ref mean   : {C_ref.abs().mean()}")

# Benchmark
NUM_WARMUPS = 5
NUM_ITERS = 10

start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]

for i in range(NUM_WARMUPS):
    bf16_matmul(A, B, C)

l2_cache_size = 1024 * 1024 * 128 # ~128MB for Blackwell
l2_cache = torch.empty(l2_cache_size // 2, dtype=torch.bfloat16, device="cuda")
assert A.device == l2_cache.device
assert B.device == l2_cache.device
assert C.device == l2_cache.device

for i in range(NUM_ITERS):
    l2_cache.random_(0, 1)
    start_events[i].record()
    bf16_matmul(A, B, C)
    end_events[i].record()
torch.cuda.synchronize()

times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
avg_time = np.mean(times) * 1e-3
std_time = np.std(times) * 1e-3
flops = 2.0 * M * N * K
tflops = flops * 1e-12
gbs = 2 * (M * K + N * K + M * N) * 1e-9

print(f"Average time: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us")
print(f"Average TFLOPS: {tflops / avg_time:.2f} TFLOp/s")
print(f"Average bandwidth: {gbs / avg_time:.2f} GB/s")
