import torch
torch.random.manual_seed(42)

# Import our Python bindings
from _C import kernel

# Generate inputs
in_ = torch.randint(0, 256, (1, 1, 512), dtype=torch.uint8, device="cuda") # no torch support for fp8e8m0
tm = torch.zeros(128, 128, dtype=torch.int32, device="cuda")

# Run kernel
kernel(in_, tm)
torch.cuda.synchronize()

recon = tm[:32, :4].reshape(-1).repeat_interleave(4).reshape(-1, 4)
recon[:, 0] = (recon[:, 0] >> 0) & 0xff
recon[:, 1] = (recon[:, 1] >> 8) & 0xff
recon[:, 2] = (recon[:, 2] >> 16) & 0xff
recon[:, 3] = (recon[:, 3] >> 24) & 0xff
recon = recon.reshape(-1)

print(in_[0, 0] - recon)
print(recon)
