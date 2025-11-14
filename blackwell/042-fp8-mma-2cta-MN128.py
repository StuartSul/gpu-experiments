import numpy as np
import torch
torch.random.manual_seed(42)

# Import our Python bindings
from _C import kernel


# Matrix dimensions
M = 16384 - 128
K = 16384
N = 16384 - 128

assert M % 128 == 0
assert K % 128 == 0 and K >= 512
assert N % 128 == 0

# Generate random matrices
print('Generating inputs...')
A = (torch.randn(M, K, dtype=torch.bfloat16, device="cuda:0") / K ** 0.25).to(torch.float8_e4m3fn)
B = (torch.randn(N, K, dtype=torch.bfloat16, device="cuda:0") / K ** 0.25).to(torch.float8_e4m3fn)
C = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda:0")

# Run kernel
print('Launching kernel...')
kernel(A, B, C)
torch.cuda.synchronize()

# Check correctness
C_ref = torch.matmul(A.to(torch.bfloat16), B.T.to(torch.bfloat16))
assert C_ref.dtype == C.dtype
abs_diff = torch.abs(C_ref - C)
print(f"Max absolute difference: {abs_diff.max()}")
print(f"Mean absolute difference: {abs_diff.mean()}")

# Benchmark
NUM_WARMUPS = 5
NUM_ITERS = 10

start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_ITERS)]

for i in range(NUM_WARMUPS):
    kernel(A, B, C)

l2_cache_size = 1024 * 1024 * 128 # ~128MB for Blackwell
l2_cache = torch.randn(l2_cache_size // 2, dtype=torch.bfloat16)
cache_clear = lambda: l2_cache.random_(0, 1)

for i in range(NUM_ITERS):
    cache_clear()
    start_events[i].record()
    kernel(A, B, C)
    end_events[i].record()
torch.cuda.synchronize()

times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
avg_time = np.mean(times) * 1e-3
std_time = np.std(times) * 1e-3
flops = 2.0 * M * N * K
tflops = flops * 1e-12

print(f"Average time: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us")
print(f"Average TFLOPS: {tflops / (avg_time):.2f} TFLOp/s")
