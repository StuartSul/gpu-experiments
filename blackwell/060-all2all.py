"""
torchrun --nproc_per_node=8 60-all2all.py
"""

from datetime import timedelta
import os

import torch
import torch.distributed
torch.set_printoptions(sci_mode=False)

from _C import TKParallelTensor, all2all_s2g1 as all2all


# Flags
CHECK_CORRECTNESS = True
BENCHMARK = True
PROFILE = True


def check_diff(name, A, A_ref):
    print(f"\n{name}")
    print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
    print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
    print(f"Mean:      {A.abs().mean().item():.10f}")
    print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
    print(f"Max:       {A.abs().max().item():.10f}")
    print(f"Ref max:   {A_ref.abs().max().item():.10f}")


def all2all_ref(
    src: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1
) -> torch.Tensor:
    """
    all-to-all collective operation on a 4D tensor. Gathers on gather_idx, scatters on scatter_idx.

    Args:
        src: torch.tensor sharded on gather_idx
        scatter_idx: index to scatter
        gather_idx: index to gather

    Returns:
        torch.tensor: torch.tensor gathered on gather_idx and sharded on scatter_idx
    """

    num_ranks = torch.distributed.get_world_size()

    if scatter_idx == 2 and gather_idx == 1:
        B, N_per_rank, H, D = src.shape
        N = N_per_rank * num_ranks
        H_per_rank = H // num_ranks

        src_t = src.view(B, N_per_rank, num_ranks, H_per_rank, D).permute(2, 1, 0, 3, 4).contiguous()
        
        if num_ranks > 1:
            dst = torch.empty_like(src_t)
            torch.distributed.all_to_all_single(dst, src_t)
        else:
            dst = src_t

        return dst.permute(2, 0, 1, 3, 4).reshape(B, N_per_rank * num_ranks, H_per_rank, D)
    
    elif scatter_idx == 1 and gather_idx == 2:
        B, N, H_per_rank, D = src.shape
        H = H_per_rank * num_ranks
        N_per_rank = N // num_ranks

        src_t = src.view(B, num_ranks, N_per_rank, H_per_rank, D).permute(1, 3, 2, 0, 4).contiguous()
        
        if num_ranks > 1:
            dst = torch.empty_like(src_t)
            torch.distributed.all_to_all_single(dst, src_t)
        else:
            dst = src_t

        return dst.permute(3, 2, 0, 1, 4).reshape(B, N_per_rank, H_per_rank * num_ranks, D)
    
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


# Parameters
N = 131072
H = 128
D = 128

# Initialize distributed environment
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
world_size = int(os.environ.get("WORLD_SIZE", 1))
assert world_size == local_world_size, "multi-node runs are not supported"
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
torch.distributed.init_process_group(
    backend="nccl",
    device_id=local_rank,
    rank=rank,
    world_size=world_size,
    timeout=timedelta(seconds=30),
)
torch.random.manual_seed(rank + 42)

# Print global parameters
if rank == 0:
    print(f"N: {N}, H: {H}, D: {D}")

if CHECK_CORRECTNESS:
    # Allocate tensors
    src = TKParallelTensor((1, N // world_size, H, D), dtype=torch.bfloat16, local_rank=local_rank, local_world_size=local_world_size, multicast=False)
    dst = TKParallelTensor((1, N, H // world_size, D), dtype=torch.bfloat16, local_rank=local_rank, local_world_size=local_world_size, multicast=False)

    # Generate inputs
    torch.randn((1, N // world_size, H, D), out=src.data_)
    torch.zeros((1, N, H // world_size, D), out=dst.data_)

    # Run PyTorch reference
    dst_ref = all2all_ref(src.data_, scatter_idx=2, gather_idx=1)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    # Run kernel
    all2all(dst, src)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    # Check correctness
    for i in range(world_size):
        if i == rank:
            check_diff(f"Rank {rank}", dst.data_, dst_ref)
        torch.distributed.barrier()

if BENCHMARK:
    # Allocate tensors
    src = TKParallelTensor((1, N // world_size, H, D), dtype=torch.bfloat16, local_rank=local_rank, local_world_size=local_world_size, multicast=False)
    dst = TKParallelTensor((1, N, H // world_size, D), dtype=torch.bfloat16, local_rank=local_rank, local_world_size=local_world_size, multicast=False)

    # Generate inputs
    torch.randn((1, N // world_size, H, D), out=src.data_)
    torch.zeros((1, N, H // world_size, D), out=dst.data_)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    chunk_size = (N // world_size) * (H // world_size) * D * 2 # chunk sent from one rank to another
    per_rank_comm_size = chunk_size * (world_size - 1)
    total_comm_size = per_rank_comm_size * world_size # N * (N - 1) * chunk_size

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_WARMUPS):
        all2all_ref(src.data_, scatter_idx=2, gather_idx=1)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        all2all_ref(src.data_, scatter_idx=2, gather_idx=1)
    end_event.record()
    torch.distributed.barrier()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) * 1e-3
    avg_time = total_time / NUM_ITERS

    if rank == 0:
        print("\nB200 Max Unidirectional NVL Bandwidth: 900 GB/s")
        print(f"Per-rank data sent: {per_rank_comm_size * 1e-9:.2f} GB")

    for i in range(world_size):
        if i == rank:
            print(f'Time taken: {avg_time * 1e6:.2f} us')
            print(f'Unidirectional Bandwidth, USP: {per_rank_comm_size * 1e-9 / avg_time:.2f} GB/s')
        torch.distributed.barrier()

    for i in range(NUM_WARMUPS):
        all2all(dst, src)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        all2all(dst, src)
    end_event.record()
    torch.distributed.barrier()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) * 1e-3
    avg_time = total_time / NUM_ITERS

    if rank == 0:
        print("\nB200 Max Unidirectional NVL Bandwidth: 900 GB/s")
        print(f"Per-rank data sent: {per_rank_comm_size * 1e-9:.2f} GB")

    for i in range(world_size):
        if i == rank:
            print(f'Time taken: {avg_time * 1e6:.2f} us')
            print(f'Unidirectional Bandwidth, USP: {per_rank_comm_size * 1e-9 / avg_time:.2f} GB/s')
        torch.distributed.barrier()
    torch.cuda.synchronize()

# Profile
if PROFILE:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        with_stack=True
    ) as profiler:
        # Allocate tensors
        src = TKParallelTensor((1, N // world_size, H, D), dtype=torch.bfloat16, local_rank=local_rank, local_world_size=local_world_size, multicast=False)
        dst = TKParallelTensor((1, N, H // world_size, D), dtype=torch.bfloat16, local_rank=local_rank, local_world_size=local_world_size, multicast=False)

        # Generate inputs
        torch.randn((1, N // world_size, H, D), out=src.data_)
        torch.zeros((1, N, H // world_size, D), out=dst.data_)

        # Profile
        for i in range(NUM_WARMUPS + NUM_ITERS):
            all2all(dst, src)

    # Export to Chrome trace format
    profiler.export_chrome_trace(f"all2all_rank{rank}.json")
    if rank == 0:
        print(f"\nProfiler trace exported to all2all_rankN.json")

# Clean up
torch.distributed.destroy_process_group()
