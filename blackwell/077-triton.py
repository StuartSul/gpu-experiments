"""
Trying to learn the basics...

Vector addition example from official docs, with my comments and timings

Some terminologies:
- Triton "program" = CUDA threadblock
- Triton "block size" != CUDA blockDim (number of threads per block)
    - Block size is logical; triton internally decides how to divide the work
    - Block size must be powers of 2

JIT takes about 260 ms for this kernel, and about 80 us for cached version
"""

import time

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
print(DEVICE, type(DEVICE))

@triton.jit
def add_kernel(
    x_ptr, # x, y, output should have the same size (n_elments)
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr # number of elements each "program" should process (= CUDA threadblock)
):
    # Retrieve proper index
    # 1 thread per 1 element, apparently
    pid = tl.program_id(axis=0) # assume 1D grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask guards OOB access
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    n_elements = output.numel()

    # In triton, grid is defined as either tuple or callable
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),) # I think it should return a tuple

    # Call the kernel!
    # Grid is passed in as "index" to the jit'ed kernels
    # Torch.tensor objects are implicitly converted to pointers
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


if __name__ == '__main__':
    torch.manual_seed(0)
    size = 9832322

    # I'm guessing triton implicitly guesses the types?
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = torch.empty_like(x)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    add(x, y, output_triton)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}')

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    add(x, y, output_triton)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    print(f"First call time (with JIT): {(t1 - t0)*1e3:.3f} ms")
    print(f"Second call time (cached): {(t3 - t2)*1e3:.3f} ms")

    # do_bench vs cuda events
    avg_time_triton = triton.testing.do_bench(lambda: add(x, y, output_triton), warmup=5, rep=15, return_mode="mean")
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(1000):
        add(x, y, output_triton)
    end.record()
    end.synchronize()
    avg_time_cuda_events = start.elapsed_time(end) / 1000

    print(avg_time_triton)
    print(avg_time_cuda_events)
