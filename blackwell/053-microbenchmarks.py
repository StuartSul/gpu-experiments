import torch

from _C import kernel

# Blackwell
print(f"Clock rate: 120 MHz, {1000 / 120:.2f} ns per cycle")

M = 16384
N = 16384

A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

kernel(A)
torch.cuda.synchronize()
