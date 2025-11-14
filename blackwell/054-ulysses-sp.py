# Run with torchrun --nproc_per_node=8 54-ulysses-sp.py

import time
import sys

import numpy as np
import torch
torch.set_printoptions(sci_mode=False)

from flash_attn.cute.interface import FlashAttnFunc

# Global parameters
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
H = 128
D_qk = 192
D_vo = 128

# Config
CHECK_CORRECTNESS = False
BENCHMARK = True
NUM_WARMUPS = 5
NUM_ITERS = 10

# Distributed setup
torch.distributed.init_process_group(backend='nccl')
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
device = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)
torch.random.manual_seed(rank + 42)

# Print global parameters
if rank == 0:
    print(f"N: {N}, H: {H}, D_qk: {D_qk}, D_vo: {D_vo}")

# Generate inputs
Q = torch.randn(1, N // world_size, H, D_qk, dtype=torch.bfloat16, device=device)
K = torch.randn(1, N // world_size, H, D_qk, dtype=torch.bfloat16, device=device)
V = torch.randn(1, N // world_size, H, D_vo, dtype=torch.bfloat16, device=device)
softmax_scale = 1 / (D_qk ** 0.5)


def all_to_all_4D(
    input: torch.tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group: torch.distributed.ProcessGroup | None = None,
    sync: bool = False
) -> torch.tensor:
    """
    all-to-all collective operation on a 4D tensor. Gathers on gather_idx, scatters on scatter_idx.

    Args:
        input: torch.tensor sharded on gather_idx
        scatter_idx: index to scatter
        gather_idx: index to gather
        group: torch process group
        sync: whether to synchronize after all-to-all

    Returns:
        torch.tensor: torch.tensor gathered on gather_idx and sharded on scatter_idx
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    num_ranks = torch.distributed.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        B, N_per_rank, H, D = input.shape
        N = N_per_rank * num_ranks
        H_per_rank = H // num_ranks

        input_t = input.view(B, N_per_rank, num_ranks, H_per_rank, D).permute(2, 1, 0, 3, 4).contiguous()
        
        if num_ranks > 1:
            output = torch.empty_like(input_t)
            torch.distributed.all_to_all_single(output, input_t, group=group)
            if sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        return output.permute(2, 0, 1, 3, 4).reshape(B, N_per_rank * num_ranks, H_per_rank, D)
    
    elif scatter_idx == 1 and gather_idx == 2:
        B, N, H_per_rank, D = input.shape
        H = H_per_rank * num_ranks
        N_per_rank = N // num_ranks

        input_t = input.view(B, num_ranks, N_per_rank, H_per_rank, D).permute(1, 3, 2, 0, 4).contiguous()
        
        if num_ranks > 1:
            output = torch.empty_like(input_t)
            torch.distributed.all_to_all_single(output, input_t, group=group)
            if sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        return output.permute(3, 2, 0, 1, 4).reshape(B, N_per_rank, H_per_rank * num_ranks, D)
    
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


def pytorch_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        S = Q.permute(0, 2, 1, 3) @ K.permute(0, 2, 3, 1) * softmax_scale
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float('-inf'))
        P = S.softmax(dim=-1)
        O = P @ V.permute(0, 2, 1, 3)
        return O.permute(0, 2, 1, 3)


def flash_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        O, _ = FlashAttnFunc.apply(Q, K, V, softmax_scale, True) # always causal
        return O


def ulysses_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        Q = all_to_all_4D(Q, 2, 1)
        K = all_to_all_4D(K, 2, 1)
        V = all_to_all_4D(V, 2, 1)
        O, _ = FlashAttnFunc.apply(Q, K, V, softmax_scale, True) # always causal
        O = all_to_all_4D(O, 1, 2)
        return O


# Check correctness
if CHECK_CORRECTNESS:
    # Diff checking utility
    def check_diff(name, A, A_ref):
        print(f"\n{name}")
        print(f"Max diff:  {((A - A_ref).abs().max().item()):.10f}")
        print(f"Mean diff: {((A - A_ref).abs().mean().item()):.10f}")
        print(f"Mean:      {A.abs().mean().item():.10f}")
        print(f"Ref mean:  {A_ref.abs().mean().item():.10f}")
        print(f"Max:       {A.abs().max().item():.10f}")
        print(f"Ref max:   {A_ref.abs().max().item():.10f}")

    # Ulysses MHA
    O = ulysses_attn_fwd(Q, K, V)

    if rank == 0:
        Q_gathered = [torch.empty_like(Q) for _ in range(world_size)]
        K_gathered = [torch.empty_like(K) for _ in range(world_size)]
        V_gathered = [torch.empty_like(V) for _ in range(world_size)]
        O_gathered = [torch.empty_like(O) for _ in range(world_size)]
    else:
        Q_gathered = None
        K_gathered = None
        V_gathered = None
        O_gathered = None

    torch.distributed.gather(Q, Q_gathered, 0)
    torch.distributed.gather(K, K_gathered, 0)
    torch.distributed.gather(V, V_gathered, 0)
    torch.distributed.gather(O, O_gathered, 0)

    if rank == 0:
        Q_full = torch.cat(Q_gathered, dim=1)
        K_full = torch.cat(K_gathered, dim=1)
        V_full = torch.cat(V_gathered, dim=1)
        O_full = torch.cat(O_gathered, dim=1)

        # Flash MHA
        O_flash = flash_attn_fwd(Q_full, K_full, V_full)

        # Pytorch MHA
        O_ref = pytorch_attn_fwd(Q_full, K_full, V_full)

        # Check results
        check_diff("O (flash)", O_flash, O_ref)
        check_diff("O (ulysses)", O_full, O_ref)

# Benchmark
if BENCHMARK:
    l2_cache = torch.empty(1024 * 1024 * 64, dtype=torch.bfloat16, device=device)

    for i in range(NUM_WARMUPS):
        ulysses_attn_fwd(Q, K, V)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    times = []

    for i in range(NUM_ITERS):
        l2_cache.random_()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ulysses_attn_fwd(Q, K, V)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = np.mean(times)
    std_time = np.std(times)
    flops = 2 * H * ((N / world_size) * ((N / world_size) + 1) / 2) * (D_qk + D_vo)
    tflops = flops * 1e-12

    for i in range(world_size):
        if i == rank:
            print(f'Time taken: {avg_time * 1e6:.2f} ± {std_time * 1e6:.2f} us')
            print(f'TFLOPS: {tflops / avg_time:.2f} TFLOP/s')
        torch.distributed.barrier()

# Clean up
torch.distributed.destroy_process_group()
