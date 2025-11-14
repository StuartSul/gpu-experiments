import torch

from _C import kernel

t = torch.zeros(128, 128, dtype=torch.int32, device="cuda")

kernel(t)
torch.cuda.synchronize()

print(t[:, :4])
