"""
Another bit of hands-on for myself

Some more notes
- Triton infers device from tensors passed in. All tensors passed to triton kernels must be on the same device
"""

import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device() # torch.device class
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index) # Python dict[str, int]
NUM_SM = properties["multiprocessor_count"] # number of SMs (148 for B200)
NUM_REG = properties["max_num_regs"] # number of registers per SM (65536 for B200; 32-bit registers)
SIZE_SMEM = properties["max_shared_mem"] # not sure why, but 232,448B for B200 (maybe some are reserved for Triton)
WARP_SIZE = properties["warpSize"] # threads per warp; always 32, as we know
target = triton.runtime.driver.active.get_current_target() # gives metadata about the device (vendor, compute capability, warp size)


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr
):
    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(axis=0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max) # __expf
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4

    y = torch.empty_like(x)

    # Pre-compile kernel
    kernel = softmax_kernel.warmup(
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1,)
    )
    kernel._init_handles()
    
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy = NUM_REG // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy # I think this is trying to do permanent grid
    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(18231, 4723, device=DEVICE)
    y_pytorch = torch.softmax(x, axis=1)
    y_triton = softmax(x)
    assert torch.allclose(y_pytorch, y_triton)
