import torch

from _C import kernel


A = torch.zeros(128 + 2, device="cuda", dtype=torch.int32)
kernel(A)
torch.cuda.synchronize()
print(A)
