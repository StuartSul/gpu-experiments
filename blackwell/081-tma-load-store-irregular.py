from time import perf_counter
import torch

from _C import kernel


M = 16384
N = 16384

t_in = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
t_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

kernel(t_in, t_out)
torch.cuda.synchronize()

breakpoint()
