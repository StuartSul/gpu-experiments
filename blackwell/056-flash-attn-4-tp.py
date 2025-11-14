# Run with torchrun --nproc_per_node=8 56-flash-attn-4-tp.py

import time
import sys

import numpy as np
import torch
torch.set_printoptions(sci_mode=False)

from flash_attn.cute.interface import FlashAttnFunc

# Global parameters
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
H = 128
D_qk = 192
D_vo = 128

# Config
CHECK_CORRECTNESS = False
BENCHMARK = True
NUM_WARMUPS = 5
NUM_ITERS = 10

# Distributed setup
torch.distributed.init_process_group(backend='nccl')
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
torch.random.manual_seed(rank + 42)

# Print global parameters
if rank == 0:
    print(f"N: {N}, H: {H}, D_qk: {D_qk}, D_vo: {D_vo}")

# Generate inputs
Q = torch.randn(1, N, H // world_size, D_qk, dtype=torch.bfloat16, device=device)
K = torch.randn(1, N, H // world_size, D_qk, dtype=torch.bfloat16, device=device)
V = torch.randn(1, N, H // world_size, D_vo, dtype=torch.bfloat16, device=device)
softmax_scale = 1 / (D_qk ** 0.5)


def flash_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        O, _ = FlashAttnFunc.apply(Q, K, V, softmax_scale, True) # always causal
        torch.distributed.all_reduce(O) # not exactly correct, but serves as benchmark
        return O


# Benchmark
l2_cache = torch.empty(1024 * 1024 * 64, dtype=torch.bfloat16, device=device)

for i in range(NUM_WARMUPS):
    flash_attn_fwd(Q, K, V)
torch.distributed.barrier()
torch.cuda.synchronize()

times = []

for i in range(NUM_ITERS):
    l2_cache.random_()
    torch.distributed.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    flash_attn_fwd(Q, K, V)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg_time = np.mean(times)
std_time = np.std(times)
flops = 2 * (H / world_size) * (N * (N + 1) / 2) * (D_qk + D_vo)
tflops = flops * 1e-12

for i in range(world_size):
    if i == rank:
        print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
        print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')
    torch.distributed.barrier()

# Clean up
torch.distributed.destroy_process_group()
