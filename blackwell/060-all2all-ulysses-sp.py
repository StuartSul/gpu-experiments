"""
A separate file for testing the ulysses-style SP with all2all kernel
torchrun --nproc_per_node=8 60-all2all-ulysses-sp.py
"""

from datetime import timedelta
import os

import torch
import torch.distributed
torch.set_printoptions(sci_mode=False)

import cuda.bindings.driver as cuda
import cutlass
from cutlass.cute.runtime import from_dlpack
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

from _C import TKParallelTensor, all2all_s2g1, all2all_s1g2


# Flags
CHECK_CORRECTNESS = True
BENCHMARK = True
PROFILE = True

# Parameters
B = 1
N = 131072
H = 128
D_qk = 192
D_vo = 128


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


compile_funcs = {}
def flash_attn_fwd_raw(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    softmax_scale: float,
    O: torch.Tensor | None = None,
    L: torch.Tensor | None = None
) -> torch.Tensor:
    B, N, H, D_qk = Q.shape
    _, _, _, D_vo = V.shape

    if O is None:
        O = torch.empty(B, N, H, D_vo, dtype=Q.dtype, device=Q.device)
    if L is None:
        L = torch.empty(B, H, N, dtype=torch.float32, device=Q.device)

    Q_cute = from_dlpack(Q.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=Q.ndim - 1)
    K_cute = from_dlpack(K.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=K.ndim - 1)
    V_cute = from_dlpack(V.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim - 1)
    O_cute = from_dlpack(O.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim - 1)
    L_cute = from_dlpack(L.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=L.ndim - 1)

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Arbitrary key to use the global dict
    if "FA4" not in compile_funcs:
        _fa_fwd = FlashAttentionForwardSm100(
            D_qk,
            D_vo,
            qhead_per_kvhead=1,
            is_causal=True,
            is_local=False,
            pack_gqa=False,
            is_persistent=False # always false if causal
        )
        compile_funcs["FA4"] = cutlass.cute.compile(
            _fa_fwd, Q_cute, K_cute, V_cute, O_cute, L_cute, softmax_scale, current_stream,
            None, None, None, None, None, None, None, None, None
        )

    compile_funcs["FA4"](
        Q_cute, K_cute, V_cute, O_cute, L_cute, softmax_scale, current_stream,
        None, None, None, None, None, None, None, None, None
    )

    return O, L


def ulysses_attn_fwd_ref(
    Q_sp: torch.Tensor,
    K_sp: torch.Tensor,
    V_sp: torch.Tensor
) -> torch.Tensor:
    Q_tp_ref = all2all_ref(Q_sp, 2, 1)
    K_tp_ref = all2all_ref(K_sp, 2, 1)
    V_tp_ref = all2all_ref(V_sp, 2, 1)
    O_tp_ref, _ = flash_attn_fwd_raw(Q_tp_ref, K_tp_ref, V_tp_ref, softmax_scale)
    O_sp_ref = all2all_ref(O_tp_ref, 1, 2)
    return O_sp_ref


def ulysses_attn_fwd(
    Q_sp: TKParallelTensor,
    Q_tp: TKParallelTensor,
    K_sp: TKParallelTensor,
    K_tp: TKParallelTensor,
    V_sp: TKParallelTensor,
    V_tp: TKParallelTensor,
    O_sp: TKParallelTensor,
    O_tp: TKParallelTensor,
    L: torch.Tensor
) -> torch.Tensor:
    all2all_s2g1(Q_tp, Q_sp)
    all2all_s2g1(K_tp, K_sp)
    all2all_s2g1(V_tp, V_sp)
    torch.distributed.barrier() # TODO: remove
    flash_attn_fwd_raw(Q_tp, K_tp, V_tp, softmax_scale, O_tp, L)
    torch.distributed.barrier() # TODO: remove
    all2all_s1g2(O_sp, O_sp_ipc_ptrs, O_tp, O_tp_ipc_ptrs, broker)
    return O_sp


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

# Initialize KittensBroker
broker = KittensBroker(local_rank, local_world_size)

# Generate inputs
Q_sp = torch.randn(B, N // world_size, H, D_qk, dtype=torch.bfloat16, device=device)
Q_tp = torch.zeros(B, N, H // world_size, D_qk, dtype=torch.bfloat16, device=device)
K_sp = torch.randn(B, N // world_size, H, D_qk, dtype=torch.bfloat16, device=device)
K_tp = torch.zeros(B, N, H // world_size, D_qk, dtype=torch.bfloat16, device=device)
V_sp = torch.randn(B, N // world_size, H, D_vo, dtype=torch.bfloat16, device=device)
V_tp = torch.zeros(B, N, H // world_size, D_vo, dtype=torch.bfloat16, device=device)
O_sp = torch.zeros(B, N // world_size, H, D_vo, dtype=torch.bfloat16, device=device)
O_tp = torch.zeros(B, N, H // world_size, D_vo, dtype=torch.bfloat16, device=device)
L = torch.zeros(B, H, N, dtype=torch.float32, device=device)
softmax_scale = 1 / (D_qk ** 0.5)

# Generate IPC pointers
Q_sp_ipc_ptrs = broker.gather_ipc_ptrs(Q_sp)
Q_tp_ipc_ptrs = broker.gather_ipc_ptrs(Q_tp)
K_sp_ipc_ptrs = broker.gather_ipc_ptrs(K_sp)
K_tp_ipc_ptrs = broker.gather_ipc_ptrs(K_tp)
V_sp_ipc_ptrs = broker.gather_ipc_ptrs(V_sp)
V_tp_ipc_ptrs = broker.gather_ipc_ptrs(V_tp)
O_sp_ipc_ptrs = broker.gather_ipc_ptrs(O_sp)
O_tp_ipc_ptrs = broker.gather_ipc_ptrs(O_tp)

if CHECK_CORRECTNESS:
    O_ref = ulysses_attn_fwd_ref(Q_sp, K_sp, V_sp)
    O_ulysses = ulysses_attn_fwd(
        Q_sp, Q_sp_ipc_ptrs, Q_tp, Q_tp_ipc_ptrs,
        K_sp, K_sp_ipc_ptrs, K_tp, K_tp_ipc_ptrs,
        V_sp, V_sp_ipc_ptrs, V_tp, V_tp_ipc_ptrs,
        O_sp, O_sp_ipc_ptrs, O_tp, O_tp_ipc_ptrs,
        L
    )

    for i in range(world_size):
        if i == rank:
            check_diff("O", O_ref, O_ulysses)
        torch.distributed.barrier()



quit()

if BENCHMARK:
    for i in range(NUM_WARMUPS):
        ulysses_attn_fwd(Q_sp, Q_sp_ipc_ptrs, Q_tp, Q_tp_ipc_ptrs, K_sp, K_sp_ipc_ptrs, K_tp, K_tp_ipc_ptrs, V_sp, V_sp_ipc_ptrs, V_tp, V_tp_ipc_ptrs, O_sp, O_sp_ipc_ptrs, O_tp, O_tp_ipc_ptrs, L)
    torch.distributed.barrier()
    torch.cuda.synchronize()






















# Print global parameters
if rank == 0:
    print(f"N: {N}, H: {H}, D: {D}")

if CHECK_CORRECTNESS:
    # Generate inputs
    src = torch.randn(1, N // world_size, H, D, dtype=torch.bfloat16, device=device)
    dst = torch.zeros(1, N, H // world_size, D, dtype=torch.bfloat16, device=device)

    # Initialize KittensIPC pointers
    src_ipc_ptrs = broker.gather_ipc_ptrs(src)
    dst_ipc_ptrs = broker.gather_ipc_ptrs(dst)

    # Run PyTorch reference
    dst_ref = all2all_ref(src, scatter_idx=2, gather_idx=1)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    # Run kernel
    all2all_s2g1(dst, dst_ipc_ptrs, src, src_ipc_ptrs, broker)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    # Check correctness
    for i in range(world_size):
        if i == rank:
            check_diff(f"Rank {rank}", dst, dst_ref)
        torch.distributed.barrier()

if BENCHMARK:
    # Generate inputs
    src = torch.randn(1, N // world_size, H, D, dtype=torch.bfloat16, device=device)
    dst = torch.zeros(1, N, H // world_size, D, dtype=torch.bfloat16, device=device)

    # Initialize KittensIPC pointers
    src_ipc_ptrs = broker.gather_ipc_ptrs(src)
    dst_ipc_ptrs = broker.gather_ipc_ptrs(dst)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    chunk_size = (N // world_size) * (H // world_size) * D * 2 # chunk sent from one rank to another
    per_rank_comm_size = chunk_size * (world_size - 1)
    total_comm_size = per_rank_comm_size * world_size # N * (N - 1) * chunk_size

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_WARMUPS):
        all2all_ref(src, scatter_idx=2, gather_idx=1)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        all2all_ref(src, scatter_idx=2, gather_idx=1)
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
        all2all_s2g1(dst, dst_ipc_ptrs, src, src_ipc_ptrs, broker)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        all2all_s2g1(dst, dst_ipc_ptrs, src, src_ipc_ptrs, broker)
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
    # Generate inputs
    src = torch.randn(1, N // world_size, H, D, dtype=torch.bfloat16, device=device)
    dst = torch.zeros(1, N, H // world_size, D, dtype=torch.bfloat16, device=device)

    # Initialize KittensIPC pointers
    src_ipc_ptrs = broker.gather_ipc_ptrs(src)
    dst_ipc_ptrs = broker.gather_ipc_ptrs(dst)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        with_stack=True
    ) as profiler:
        for i in range(NUM_WARMUPS + NUM_ITERS):
            all2all_s2g1(dst, dst_ipc_ptrs, src, src_ipc_ptrs, broker)

    # Export to Chrome trace format
    profiler.export_chrome_trace(f"all2all_rank{rank}.json")
    if rank == 0:
        print(f"\nProfiler trace exported to all2all_rankN.json")

# Benchmark Ulysses-style sequence parallelism
if BENCHMARK_USP:

    def ulysses_attn_fwd_ref() -> torch.Tensor:
        with torch.no_grad():
            Q_tp_ref = all2all_ref(Q_sp, 2, 1)
            K_tp_ref = all2all_ref(K_sp, 2, 1)
            V_tp_ref = all2all_ref(V_sp, 2, 1)
            O_tp_ref, _ = flash_attn_fwd_raw(Q_tp_ref, K_tp_ref, V_tp_ref, softmax_scale)
            O_sp_ref = all2all_ref(O_tp_ref, 1, 2)
            return O_sp_ref

    def ulysses_attn_fwd() -> torch.Tensor:
        with torch.no_grad():
            all2all_s2g1(Q_tp, Q_tp_ipc_ptrs, Q_sp, Q_sp_ipc_ptrs, broker)
            all2all_s2g1(K_tp, K_tp_ipc_ptrs, K_sp, K_sp_ipc_ptrs, broker)
            all2all_s2g1(V_tp, V_tp_ipc_ptrs, V_sp, V_sp_ipc_ptrs, broker)
            flash_attn_fwd_raw(Q_tp, K_tp, V_tp, softmax_scale, O_tp, L_tp)
            all2all_s1g2(O_sp, O_tp_ipc_ptrs, O_tp, Q_tp_ipc_ptrs, broker)
            return O_sp

    # Generate inputs
    Q_sp = torch.randn(1, N // world_size, H, D, dtype=torch.bfloat16, device=device)
    Q_tp = torch.zeros(1, N, H // world_size, D, dtype=torch.bfloat16, device=device)
    K_sp = torch.randn(1, N // world_size, H, D, dtype=torch.bfloat16, device=device)
    K_tp = torch.zeros(1, N, H // world_size, D, dtype=torch.bfloat16, device=device)
    V_sp = torch.randn(1, N // world_size, H, D, dtype=torch.bfloat16, device=device)
    V_tp = torch.zeros(1, N, H // world_size, D, dtype=torch.bfloat16, device=device)
    O_sp = torch.zeros(1, N // world_size, H, D, dtype=torch.bfloat16, device=device)
    O_tp = torch.zeros(1, N, H // world_size, D, dtype=torch.bfloat16, device=device)
    L_sp = torch.zeros(1, H, N, dtype=torch.float32, device=device)
    L_tp = torch.zeros(1, H, N, dtype=torch.float32, device=device)

    # Initialize KittensIPC pointers
    Q_sp_ipc_ptrs = broker.gather_ipc_ptrs(Q_sp)
    Q_tp_ipc_ptrs = broker.gather_ipc_ptrs(Q_tp)
    K_sp_ipc_ptrs = broker.gather_ipc_ptrs(K_sp)
    K_tp_ipc_ptrs = broker.gather_ipc_ptrs(K_tp)
    V_sp_ipc_ptrs = broker.gather_ipc_ptrs(V_sp)
    V_tp_ipc_ptrs = broker.gather_ipc_ptrs(V_tp)
    O_sp_ipc_ptrs = broker.gather_ipc_ptrs(O_sp)
    O_tp_ipc_ptrs = broker.gather_ipc_ptrs(O_tp)

    # Define softmax scale
    softmax_scale = 1 / (D ** 0.5)

    # Benchmark
    NUM_WARMUPS = 5
    NUM_ITERS = 10

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i in range(NUM_WARMUPS):
        ulysses_attn_fwd_ref()
    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        ulysses_attn_fwd_ref()
    end_event.record()
    torch.distributed.barrier()
    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event) * 1e-3
    avg_time = total_time / NUM_ITERS
    flops = 2 * H * ((N / world_size) * ((N / world_size) + 1) / 2) * (D + D)
    tflops = flops * 1e-12

    if rank == 0:
        print("\nB200 Max Unidirectional NVL Bandwidth: 900 GB/s")
        print(f"Per-rank data sent: {per_rank_comm_size * 1e-9:.2f} GB")

    for i in range(world_size):
        if i == rank:
            print(f'Time taken: {avg_time * 1e6:.2f} us')
            print(f'Unidirectional Bandwidth, USP: {per_rank_comm_size * 1e-9 / avg_time:.2f} GB/s')
        torch.distributed.barrier()

    for i in range(NUM_WARMUPS):
        all2all_s2g1(dst, dst_ipc_ptrs, src, src_ipc_ptrs, broker)
    torch.distributed.barrier()
    torch.cuda.synchronize()

    start_event.record()
    for i in range(NUM_ITERS):
        all2all_s2g1(dst, dst_ipc_ptrs, src, src_ipc_ptrs, broker)
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

# Clean up
torch.distributed.destroy_process_group()
del compile_funcs
