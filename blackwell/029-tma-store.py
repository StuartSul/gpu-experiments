import torch

from _C import kernel


M = 16384
N = 16384

t_in = torch.arange(
    M * N, dtype=torch.int32, device="cuda"
).to(torch.uint16).view(torch.bfloat16).reshape(M, N)
t_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

kernel(t_in, t_out)
torch.cuda.synchronize()

t_out = t_out.view(torch.uint16).to(torch.int32)
print(t_out)
