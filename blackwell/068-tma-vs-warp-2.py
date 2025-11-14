import numpy as np
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)


from _C import kernel


# Global parameters
N = 32768

# Generate inputs
dst = torch.zeros(N, N, dtype=torch.bfloat16, device="cuda")

# Run kernel
kernel(dst)

# Check correctness
assert (dst == 3.14).all()

# Benchmark
NUM_WARMUPS = 5
NUM_ITERS = 10

for i in range(NUM_WARMUPS):
    kernel(dst)
torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for i in range(NUM_ITERS):
    kernel(dst)
end_event.record()
torch.cuda.synchronize()

total_time = start_event.elapsed_time(end_event) * 1e-3
avg_time = total_time / NUM_ITERS

total_gb = 2 * N * N * 1e-9 # must cut in half (no load)
gbps = total_gb / avg_time

print(f'Time taken: {avg_time * 1e3:.4f} ms')
print(f'Throughput: {gbps:.2f} GB/s')
