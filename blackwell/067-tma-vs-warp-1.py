import numpy as np
import torch
torch.random.manual_seed(42)
torch.set_printoptions(sci_mode=False)


from _C import kernel


# Global parameters
N = 32768

# Generate inputs
src = torch.randn(N, N, dtype=torch.bfloat16, device="cuda")
dst = torch.zeros(N, N, dtype=torch.bfloat16, device="cuda")

# Run kernel
kernel(src, dst)

# Check correctness
def check_diff(name, A, A_ref):
    print(f"\n{name}")
    print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    print(f"Mean:      {A.abs().mean().item():.10f}")
    print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
    print(f"Max:       {A.abs().max().item():.10f}")
    print(f"Ref max:   {A_ref.abs().max().item():.10f}")

check_diff("src vs dst", src, dst)

# Benchmark
NUM_WARMUPS = 5
NUM_ITERS = 10

for i in range(NUM_WARMUPS):
    kernel(src, dst)
torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for i in range(NUM_ITERS):
    kernel(src, dst)
end_event.record()
torch.cuda.synchronize()

total_time = start_event.elapsed_time(end_event) * 1e-3
avg_time = total_time / NUM_ITERS

total_gb = 2 * 2 * N * N * 1e-9
gbps = total_gb / avg_time

print(f'Time taken: {avg_time * 1e3:.4f} ms')
print(f'Throughput: {gbps:.2f} GB/s')
