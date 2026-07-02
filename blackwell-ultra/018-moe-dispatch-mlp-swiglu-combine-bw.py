"""
To run:
    make
    torchrun --nproc_per_node=4 018-moe-dispatch-mlp-swiglu-combine-bw.py
"""

import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import dispatch_mlp_swiglu_combine, schedule


NUM_LOCAL_TOKENS = 7168
HIDDEN_DIM = 7168
INTERMEDIATE_DIM = 2048
NUM_LOCAL_EXPERTS = 4
TOPK = 8
NUM_COMM_SMS = 48

WARMUP_ITERS = 5
PROFILE_ITERS = 3
TIMED_ITERS = 10


@torch.compile
def finalize(y_shared, combine_buffer, topk_weights):
    num_local_tokens, topk = topk_weights.shape
    y_routed = combine_buffer.view(num_local_tokens, topk, -1).float()
    return (y_shared.float() + (y_routed * topk_weights.unsqueeze(-1)).sum(dim=1)).to(torch.bfloat16)


def mlp_swiglu_ref(
    x_shared, x_routed, 
    w_shared_gate, w_routed_gate, 
    w_shared_up, w_routed_up, 
    w_shared_down, w_routed_down, 
    tokens_per_expert
):
    gate_shared = x_shared @ w_shared_gate.T
    up_shared = x_shared @ w_shared_up.T
    hidden_shared = (F.silu(gate_shared.float()) * up_shared.float()).to(x_shared.dtype)
    y_shared = hidden_shared @ w_shared_down.T

    gate_routed = torch.empty(x_routed.size(0), w_routed_gate.size(1), device=x_routed.device, dtype=x_routed.dtype)
    up_routed = torch.empty_like(gate_routed)
    hidden_routed = torch.empty_like(gate_routed)
    y_routed = torch.empty_like(x_routed)

    offset = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
        _x = x_routed[offset:offset + num_tokens]
        gate_routed[offset:offset + num_tokens] = _x @ w_routed_gate[expert_idx].T
        up_routed[offset:offset + num_tokens] = _x @ w_routed_up[expert_idx].T
        hidden_routed[offset:offset + num_tokens] = (F.silu(gate_routed[offset:offset + num_tokens].float()) * up_routed[offset:offset + num_tokens].float()).to(x_routed.dtype)
        y_routed[offset:offset + num_tokens] = hidden_routed[offset:offset + num_tokens] @ w_routed_down[expert_idx].T
        offset += num_tokens

    return gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

    num_experts = NUM_LOCAL_EXPERTS * world_size
    buffer_capacity = NUM_LOCAL_TOKENS * TOPK * 3 // 2

    # Generate inputs and communication buffers
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    router_logits = torch.randn(NUM_LOCAL_TOKENS, num_experts, generator=gen, device=device)
    x = symm_mem.empty(NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    x.normal_(generator=gen)
    x_handle = symm_mem.rendezvous(x, dist.group.WORLD.group_name)
    x_ptrs = [x_handle.buffer_ptrs[i] for i in range(world_size)]
    dispatch_buffer = torch.empty(buffer_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    combine_buffer = symm_mem.empty(NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    combine_buffer_handle = symm_mem.rendezvous(combine_buffer, dist.group.WORLD.group_name)
    combine_buffer_ptrs = [combine_buffer_handle.buffer_ptrs[i] for i in range(world_size)]

    # Generate weights
    w_shared_gate   = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_routed_gate   = torch.randn(NUM_LOCAL_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_shared_up     = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_routed_up     = torch.randn(NUM_LOCAL_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_shared_down   = torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM, generator=gen, device=device, dtype=torch.bfloat16) * INTERMEDIATE_DIM ** -0.5
    w_routed_down   = torch.randn(NUM_LOCAL_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM, generator=gen, device=device, dtype=torch.bfloat16) * INTERMEDIATE_DIM ** -0.5

    # Router
    topk_vals, topk_ids = torch.topk(router_logits, TOPK, dim=1)
    topk_weights = torch.softmax(topk_vals.float(), dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_ids_all = torch.empty(world_size, NUM_LOCAL_TOKENS, TOPK, dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    for _ in range(WARMUP_ITERS):
        dist.all_gather_into_tensor(topk_ids_all, topk_ids)
        schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert = schedule(
            topk_ids_all, NUM_LOCAL_EXPERTS, buffer_capacity, rank
        )
        gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, combine_buffer = dispatch_mlp_swiglu_combine(
            x, x_ptrs, dispatch_buffer, combine_buffer, combine_buffer_ptrs,
            w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
            schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert, TOPK, NUM_COMM_SMS
        )
        dist.barrier(async_op=True).block_current_stream()
        output = finalize(y_shared, combine_buffer, topk_weights)
    torch.cuda.synchronize()
    dist.barrier()

    # Trace
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(PROFILE_ITERS):
            dist.all_gather_into_tensor(topk_ids_all, topk_ids)
            schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert = schedule(
                topk_ids_all, NUM_LOCAL_EXPERTS, buffer_capacity, rank
            )
            gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, combine_buffer = dispatch_mlp_swiglu_combine(
                x, x_ptrs, dispatch_buffer, combine_buffer, combine_buffer_ptrs,
                w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
                schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert, TOPK, NUM_COMM_SMS
            )
            dist.barrier(async_op=True).block_current_stream()
            output = finalize(y_shared, combine_buffer, topk_weights)
        torch.cuda.synchronize()
    trace_path = f"trace_moe_rank{rank}.json"
    prof.export_chrome_trace(trace_path)
    if rank == 0:
        print(f"[rank {rank}] wrote {trace_path} (open in https://ui.perfetto.dev)", flush=True)
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        dist.all_gather_into_tensor(topk_ids_all, topk_ids)
        schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert = schedule(
            topk_ids_all, NUM_LOCAL_EXPERTS, buffer_capacity, rank
        )
        gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, combine_buffer = dispatch_mlp_swiglu_combine(
            x, x_ptrs, dispatch_buffer, combine_buffer, combine_buffer_ptrs,
            w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
            schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert, TOPK, NUM_COMM_SMS
        )
        dist.barrier(async_op=True).block_current_stream()
        output = finalize(y_shared, combine_buffer, topk_weights)
    end.record()
    torch.cuda.synchronize()
    dist.barrier()

    end_to_end_ms = start.elapsed_time(end) / TIMED_ITERS
    total_routed_tokens = int(tokens_per_expert.sum().item())

    # Reference implementation
    x_all = torch.empty(world_size, NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(x_all, x)
    valid = schedule_peer_rank >= 0
    x_routed_ref = torch.empty_like(dispatch_buffer)
    x_routed_ref[valid] = x_all[schedule_peer_rank[valid], schedule_peer_token_idx[valid] // TOPK]
    mlp_swiglu_refs = mlp_swiglu_ref(
        x, x_routed_ref, w_shared_gate, w_routed_gate, w_shared_up,
        w_routed_up, w_shared_down, w_routed_down, tokens_per_expert
    )
    schedule_peer_rank_all = torch.empty(world_size, buffer_capacity, dtype=torch.int32, device=device)
    schedule_peer_token_idx_all = torch.empty_like(schedule_peer_rank_all)
    y_routed_all = torch.empty(world_size, buffer_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(schedule_peer_rank_all, schedule_peer_rank)
    dist.all_gather_into_tensor(schedule_peer_token_idx_all, schedule_peer_token_idx)
    dist.all_gather_into_tensor(y_routed_all, mlp_swiglu_refs[-1])
    combine_buffer_ref = torch.empty_like(combine_buffer)
    for dst_rank in range(world_size):
        dst_valid = schedule_peer_rank_all[dst_rank] == rank
        combine_buffer_ref[schedule_peer_token_idx_all[dst_rank, dst_valid].long()] = y_routed_all[dst_rank, dst_valid]
    output_ref = finalize(mlp_swiglu_refs[-2], combine_buffer_ref, topk_weights)

    # Correctness checks for all returned tensors and final output
    names = ("gate_shared", "gate_routed", "up_shared", "up_routed", "hidden_shared", "hidden_routed", "y_shared", "y_routed", "combine_buffer", "output")
    outputs = (gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, combine_buffer, output)
    references = (*mlp_swiglu_refs, combine_buffer_ref, output_ref)
    difference_stats = []
    for name, out, ref in zip(names, outputs, references):
        if name.endswith("_routed"):
            out = out[valid]
            ref = ref[valid]
        diff = (out.float() - ref.float()).abs()
        diff_sum = diff.sum()
        diff_count = torch.tensor(diff.numel(), dtype=torch.float64, device=device)
        diff_max = diff.max()
        dist.all_reduce(diff_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(diff_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(diff_max, op=dist.ReduceOp.MAX)
        difference_stats.append((name, (diff_sum / diff_count).item(), diff_max.item()))

    flops = 6 * (NUM_LOCAL_TOKENS + total_routed_tokens) * HIDDEN_DIM * INTERMEDIATE_DIM
    stats = torch.tensor([total_routed_tokens, end_to_end_ms, flops / 1e9 / end_to_end_ms], dtype=torch.float64, device=device)
    stats_all = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(stats_all, stats)

    if rank == 0:
        print("\nMoE Dispatch + MLP SwiGLU + Combine")
        print("====================================")
        print(f"tokens/rank: {NUM_LOCAL_TOKENS}   experts: {num_experts} ({NUM_LOCAL_EXPERTS}/rank)   "
              f"topk: {TOPK}   H: {HIDDEN_DIM}   I: {INTERMEDIATE_DIM}")
        print(f"iters: warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)
        for name, diff_mean, diff_max in difference_stats:
            print(f"{name:<14} diff mean {diff_mean:.6f}   diff max {diff_max:.6f}")
        print("\nrank  routed tokens  end-to-end(ms)  MLP TFLOP/s")
        print("----  -------------  --------------  -----------")
        for rank_idx, rank_stats in enumerate(stats_all):
            routed_tokens, rank_end_to_end_ms, rank_tflops = rank_stats.tolist()
            print(f"{rank_idx:>4}  {int(routed_tokens):>13}  {rank_end_to_end_ms:>14.3f}  {rank_tflops:>11.1f}")

    dist.barrier()
    del x_handle
    del combine_buffer_handle
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
