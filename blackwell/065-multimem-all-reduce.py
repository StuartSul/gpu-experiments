import os
from typing import Callable

import torch

from _C import TKParallelTensor, tk_all_reduce


N = 16384
NUM_WARMUP_ITERS = 5
NUM_ITERS = 10


def check_diff(name: str, A: torch.Tensor, A_ref: torch.Tensor):
    clean_print(f"===============================================================================", print_once=True)
    clean_print(f"<{name}>", print_once=True)
    clean_print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    clean_print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    clean_print(f"Mean:      {A.abs().mean().item():.10f}")
    clean_print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
    clean_print(f"Max:       {A.abs().max().item():.10f}")
    clean_print(f"Ref max:   {A_ref.abs().max().item():.10f}")


def benchmark(
    func: Callable,
    num_warmup_iters: int = 10,
    num_iters: int = 50,
) -> float:
    for _ in range(num_warmup_iters):
        func()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        func()
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / num_iters

    return avg_ms


def clean_print(*args, **kwargs):
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if kwargs.pop("print_once", False):
        if local_rank == 0:
            print(*args, **kwargs)
        torch.distributed.barrier()
    else:
        for i in range(local_world_size):
            if i == local_rank:
                print(f"[Rank {i}]", *args, **kwargs)
            torch.distributed.barrier()


local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
assert world_size == local_world_size, "multi-node runs are not supported"
torch.distributed.init_process_group(
    backend="nccl",
    device_id=local_rank,
    rank=rank,
    world_size=world_size
)

device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
torch.random.manual_seed(local_rank)

tensor_tk = TKParallelTensor(
    (N, N), 
    dtype=torch.bfloat16, 
    local_rank=local_rank, 
    local_world_size=local_world_size, 
    multicast=True
)
torch.randn((N, N), out=tensor_tk.data_)
tensor_nccl = tensor_tk.data_.clone()

nccl_func = lambda: torch.distributed.all_reduce(tensor_nccl, op=torch.distributed.ReduceOp.SUM)
tk_func = lambda: tk_all_reduce(tensor_tk)

nccl_func()
tk_func()
torch.distributed.barrier() # TODO: remove this
check_diff("AllReduceSum Diff Comparison", tensor_tk.data_, tensor_nccl)

nccl_avg_ms = benchmark(nccl_func, NUM_WARMUP_ITERS, NUM_ITERS)
tk_avg_ms = benchmark(tk_func, NUM_WARMUP_ITERS, NUM_ITERS)

# Although NVLS is used, assume ring all-reduce
bytes_per_tensor = tensor_tk.data_.numel() * tensor_tk.data_.element_size()
bytes_per_channel = bytes_per_tensor * 2.0 * (local_world_size - 1) / local_world_size

nccl_gbps = ((bytes_per_channel * 1e-9) / (nccl_avg_ms * 1e-3))
tk_gbps = ((bytes_per_channel * 1e-9) / (tk_avg_ms * 1e-3))

clean_print(f"===============================================================================", print_once=True)
clean_print(f"<BF16 AllReduceSum | rank={local_rank} | world_size={local_world_size} | {N}x{N}>", print_once=True)
clean_print(f"NCCL: {nccl_avg_ms:.3f} ms | {nccl_gbps:.2f} GB/s")
clean_print(f"TK: {tk_avg_ms:.3f} ms | {tk_gbps:.2f} GB/s")

torch.distributed.destroy_process_group()
