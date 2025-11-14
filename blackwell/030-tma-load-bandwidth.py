from time import perf_counter
import torch

from _C import kernel


M = 16384
N = 16384

t_in = torch.arange(
    M * N, dtype=torch.int32, device="cuda"
).to(torch.uint16).view(torch.bfloat16).reshape(M, N)
t_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

l2_cache_size = 1024 * 1024 * 128 # ~128MB for Blackwell
l2_cache = torch.randn(l2_cache_size // 2, dtype=torch.bfloat16)
cache_clear = lambda: l2_cache.random_(0, 1)

for i in range(5):
    cache_clear()
    kernel(t_in, t_out)
    torch.cuda.synchronize()

times = []
for i in range(10):
    cache_clear()
    torch.cuda.synchronize()
    t_start = perf_counter()
    kernel(t_in, t_out)
    torch.cuda.synchronize()
    t_end = perf_counter()
    times.append(t_end - t_start)

avg_time = sum(times) / len(times)
print(f"Average time: {avg_time * 1e3:.2f} ms")
print(f"Bandwidth: {M * N * 2 / avg_time / 1e9:.2f} GB/s")
