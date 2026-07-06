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
from _C import dispatch_mlp_swiglu_combine_fwd, dispatch_mlp_swiglu_combine_bwd, schedule


NUM_LOCAL_TOKENS = 7168
HIDDEN_DIM = 7168
INTERMEDIATE_DIM = 2048
NUM_LOCAL_EXPERTS = 4
TOPK = 8
NUM_COMM_SMS = 32
MINIBATCH_SIZE = 4096
MACROBATCH_SIZE = 32 * MINIBATCH_SIZE

WARMUP_ITERS = 5
PROFILE_ITERS = 3
TIMED_ITERS = 10


@torch.compile
def finalize_fwd(y_shared, combine_buffer, topk_weights):
    num_local_tokens, topk = topk_weights.shape
    y_routed = combine_buffer.view(num_local_tokens, topk, -1).float()
    return (y_shared.float() + (y_routed * topk_weights.unsqueeze(-1)).sum(dim=1)).to(torch.bfloat16)


@torch.compile
def finalize_bwd(d_output, topk_weights, d_combine_buffer):
    num_local_tokens, topk = topk_weights.shape
    # d_combine_buffer must be symmetric memory, so we copy to it
    d_combine_buffer.view(num_local_tokens, topk, -1).copy_(d_output.float().unsqueeze(1) * topk_weights.unsqueeze(-1))
    d_y_shared = d_output  # b/c weight = 1
    return d_y_shared


def mlp_swiglu_fwd_ref(
    x_shared, x_routed,
    w_shared_gate, w_routed_gate,
    w_shared_up, w_routed_up,
    w_shared_down, w_routed_down,
    tokens_per_expert
):
    gate_shared = x_shared @ w_shared_gate.T
    up_shared = x_shared @ w_shared_up.T
    hidden_shared = (F.silu(gate_shared.float()) * up_shared.float()).to(torch.bfloat16)
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


def mlp_swiglu_bwd_ref(
    d_y_shared, d_y_routed,
    x_shared, x_routed,
    gate_shared, gate_routed,
    up_shared, up_routed,
    hidden_shared, hidden_routed,
    w_shared_gate, w_routed_gate,
    w_shared_up, w_routed_up,
    w_shared_down, w_routed_down,
    tokens_per_expert
):
    d_hidden_shared = d_y_shared @ w_shared_down
    _sigmoid = torch.sigmoid(gate_shared.float())
    _silu = gate_shared.float() * _sigmoid
    d_gate_shared = (d_hidden_shared.float() * up_shared.float() * (_sigmoid + _silu * (1.0 - _sigmoid))).to(torch.bfloat16)
    d_up_shared = (d_hidden_shared.float() * _silu).to(torch.bfloat16)
    d_x_shared = ((d_gate_shared @ w_shared_gate).float() + (d_up_shared @ w_shared_up).float()).to(torch.bfloat16)
    d_w_shared_gate = (d_gate_shared.T.float() @ x_shared.float()).to(torch.bfloat16)
    d_w_shared_up = (d_up_shared.T.float() @ x_shared.float()).to(torch.bfloat16)
    d_w_shared_down = (d_y_shared.T.float() @ hidden_shared.float()).to(torch.bfloat16)

    d_hidden_routed = torch.empty_like(gate_routed)
    d_gate_routed = torch.empty_like(gate_routed)
    d_up_routed = torch.empty_like(gate_routed)
    d_x_routed = torch.empty_like(x_routed)
    d_w_routed_gate = torch.empty_like(w_routed_gate)
    d_w_routed_up = torch.empty_like(w_routed_up)
    d_w_routed_down = torch.empty_like(w_routed_down)

    offset = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
        d_hidden_routed[offset:offset + num_tokens] = d_y_routed[offset:offset + num_tokens] @ w_routed_down[expert_idx]
        _sigmoid = torch.sigmoid(gate_routed[offset:offset + num_tokens].float())
        _silu = gate_routed[offset:offset + num_tokens].float() * _sigmoid
        d_gate_routed[offset:offset + num_tokens] = (d_hidden_routed[offset:offset + num_tokens].float() * up_routed[offset:offset + num_tokens].float() * (_sigmoid + _silu * (1.0 - _sigmoid))).to(torch.bfloat16)
        d_up_routed[offset:offset + num_tokens] = (d_hidden_routed[offset:offset + num_tokens].float() * _silu).to(torch.bfloat16)
        d_x_routed[offset:offset + num_tokens] = ((d_gate_routed[offset:offset + num_tokens] @ w_routed_gate[expert_idx]).float() + (d_up_routed[offset:offset + num_tokens] @ w_routed_up[expert_idx]).float()).to(torch.bfloat16)
        d_w_routed_gate[expert_idx] = (d_gate_routed[offset:offset + num_tokens].T.float() @ x_routed[offset:offset + num_tokens].float()).to(torch.bfloat16)
        d_w_routed_up[expert_idx] = (d_up_routed[offset:offset + num_tokens].T.float() @ x_routed[offset:offset + num_tokens].float()).to(torch.bfloat16)
        d_w_routed_down[expert_idx] = (d_y_routed[offset:offset + num_tokens].T.float() @ hidden_routed[offset:offset + num_tokens].float()).to(torch.bfloat16)
        offset += num_tokens

    return (d_x_shared, d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed, d_hidden_shared, d_hidden_routed,
            d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

    num_experts = NUM_LOCAL_EXPERTS * world_size
    schedule_capacity = NUM_LOCAL_TOKENS * TOPK * max(2, world_size // 4)

    # Generate inputs and communication buffers
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    router_logits = torch.randn(NUM_LOCAL_TOKENS, num_experts, generator=gen, device=device)
    x = symm_mem.empty(NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    x.normal_(generator=gen)
    x_handle = symm_mem.rendezvous(x, dist.group.WORLD.group_name)
    x_ptrs = [x_handle.buffer_ptrs[i] for i in range(world_size)]
    d_x = symm_mem.empty(NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    d_x_handle = symm_mem.rendezvous(d_x, dist.group.WORLD.group_name)
    d_x_ptrs = [d_x_handle.buffer_ptrs[i] for i in range(world_size)]
    combine_buffer = symm_mem.empty(NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    combine_buffer_handle = symm_mem.rendezvous(combine_buffer, dist.group.WORLD.group_name)
    combine_buffer_ptrs = [combine_buffer_handle.buffer_ptrs[i] for i in range(world_size)]
    d_combine_buffer = symm_mem.empty(NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    d_combine_buffer_handle = symm_mem.rendezvous(d_combine_buffer, dist.group.WORLD.group_name)
    d_combine_buffer_ptrs = [d_combine_buffer_handle.buffer_ptrs[i] for i in range(world_size)]

    # Generate weights and the output gradient
    w_shared_gate   = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_routed_gate   = torch.randn(NUM_LOCAL_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_shared_up     = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_routed_up     = torch.randn(NUM_LOCAL_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_shared_down   = torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM, generator=gen, device=device, dtype=torch.bfloat16) * INTERMEDIATE_DIM ** -0.5
    w_routed_down   = torch.randn(NUM_LOCAL_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM, generator=gen, device=device, dtype=torch.bfloat16) * INTERMEDIATE_DIM ** -0.5
    d_output        = torch.randn(NUM_LOCAL_TOKENS, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5

    # Router
    topk_vals, topk_ids = torch.topk(router_logits, TOPK, dim=1)
    topk_weights = torch.softmax(topk_vals.float(), dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_ids_all = torch.empty(world_size, NUM_LOCAL_TOKENS, TOPK, dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    def run_fwd_once():
        dist.all_gather_into_tensor(topk_ids_all, topk_ids)
        schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert = schedule(
            topk_ids_all, NUM_LOCAL_EXPERTS, schedule_capacity, rank
        )
        x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed = dispatch_mlp_swiglu_combine_fwd(
            x, x_ptrs, combine_buffer, combine_buffer_ptrs,
            w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
            schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
            TOPK, NUM_COMM_SMS, MACROBATCH_SIZE, MINIBATCH_SIZE
        )
        dist.barrier(async_op=True).block_current_stream()
        output = finalize_fwd(y_shared, combine_buffer, topk_weights)
        return (schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                x_routed, gate_shared, gate_routed, up_shared, up_routed,
                hidden_shared, hidden_routed, y_shared, y_routed, output)

    def run_bwd_once():
        d_y_shared = finalize_bwd(d_output, topk_weights, d_combine_buffer)
        dist.barrier(async_op=True).block_current_stream()
        (d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed, d_hidden_shared, d_hidden_routed, d_y_routed,
         d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down) = dispatch_mlp_swiglu_combine_bwd(
            d_y_shared, d_combine_buffer, d_combine_buffer_ptrs, d_x, d_x_ptrs, x, x_ptrs,
            w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down,
            x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed,
            schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
            TOPK, NUM_COMM_SMS, MACROBATCH_SIZE, MINIBATCH_SIZE
        )
        dist.barrier(async_op=True).block_current_stream()
        return (d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed, d_hidden_shared, d_hidden_routed, d_y_routed,
                d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down)

    # Forward benchmark
    for _ in range(WARMUP_ITERS):
        (schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
         x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, output) = run_fwd_once()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        (schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
         x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, output) = run_fwd_once()
    end.record()
    torch.cuda.synchronize()
    dist.barrier()
    fwd_ms = start.elapsed_time(end) / TIMED_ITERS

    # Backward benchmark
    for _ in range(WARMUP_ITERS):
        (d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed, d_hidden_shared, d_hidden_routed, d_y_routed,
         d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down) = run_bwd_once()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        run_bwd_once()
    end.record()
    torch.cuda.synchronize()
    dist.barrier()
    bwd_ms = start.elapsed_time(end) / TIMED_ITERS

    total_routed_tokens = int(num_tokens.item())

    # Forward reference
    x_all = torch.empty(world_size, NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(x_all, x)
    valid = schedule_peer_rank >= 0
    x_routed_ref = torch.zeros(schedule_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    x_routed_ref[valid] = x_all[schedule_peer_rank[valid], schedule_peer_token_idx[valid] // TOPK]
    mlp_swiglu_fwd_refs = mlp_swiglu_fwd_ref(
        x, x_routed_ref, w_shared_gate, w_routed_gate, w_shared_up,
        w_routed_up, w_shared_down, w_routed_down, tokens_per_expert
    )
    schedule_peer_rank_all = torch.empty(world_size, schedule_capacity, dtype=torch.int32, device=device)
    schedule_peer_token_idx_all = torch.empty_like(schedule_peer_rank_all)
    y_routed_all = torch.empty(world_size, schedule_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(schedule_peer_rank_all, schedule_peer_rank)
    dist.all_gather_into_tensor(schedule_peer_token_idx_all, schedule_peer_token_idx)
    dist.all_gather_into_tensor(y_routed_all, mlp_swiglu_fwd_refs[-1])
    combine_buffer_ref = torch.empty_like(combine_buffer)
    for dst_rank in range(world_size):
        dst_valid = schedule_peer_rank_all[dst_rank] == rank
        combine_buffer_ref[schedule_peer_token_idx_all[dst_rank, dst_valid].long()] = y_routed_all[dst_rank, dst_valid]
    output_ref = finalize_fwd(mlp_swiglu_fwd_refs[-2], combine_buffer_ref, topk_weights)

    # Backward reference
    d_combine_buffer_ref = torch.empty_like(combine_buffer)
    finalize_bwd(d_output, topk_weights, d_combine_buffer_ref)
    d_combine_buffer_all = torch.empty(world_size, NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(d_combine_buffer_all, d_combine_buffer_ref)
    d_y_routed_ref = torch.zeros(schedule_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    d_y_routed_ref[valid] = d_combine_buffer_all[schedule_peer_rank[valid], schedule_peer_token_idx[valid]]
    mlp_swiglu_bwd_refs = mlp_swiglu_bwd_ref(
        d_output, d_y_routed_ref, x, x_routed_ref, *mlp_swiglu_fwd_refs[:6],
        w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down, tokens_per_expert
    )
    d_x_routed_all = torch.empty(world_size, schedule_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(d_x_routed_all, mlp_swiglu_bwd_refs[1])
    d_x_ref = mlp_swiglu_bwd_refs[0].float()
    for dst_rank in range(world_size):
        dst_valid = schedule_peer_rank_all[dst_rank] == rank
        d_x_ref.index_add_(0, (schedule_peer_token_idx_all[dst_rank, dst_valid] // TOPK).long(), d_x_routed_all[dst_rank, dst_valid].float())
    d_x_ref = d_x_ref.to(torch.bfloat16)

    # Correctness checks for all returned tensors and final outputs. The routed buffers hold a single, last-remaining macrobatch
    num_macrobatches = max(1, (total_routed_tokens + MACROBATCH_SIZE - 1) // MACROBATCH_SIZE)
    fwd_macrobatch_start = (num_macrobatches - 1) * MACROBATCH_SIZE
    bwd_macrobatch_start = fwd_macrobatch_start if num_macrobatches == 1 else (num_macrobatches - 2) * MACROBATCH_SIZE
    names = ("gate_shared", "gate_routed", "up_shared", "up_routed", "hidden_shared", "hidden_routed", "y_shared", "y_routed", "combine_buffer", "output",
             "d_x", "d_x_routed", "d_gate_shared", "d_gate_routed", "d_up_shared", "d_up_routed", "d_hidden_shared", "d_hidden_routed", "d_y_routed",
             "d_w_shared_gate", "d_w_routed_gate", "d_w_shared_up", "d_w_routed_up", "d_w_shared_down", "d_w_routed_down")
    outputs = (gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed, combine_buffer, output,
               d_x, d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed, d_hidden_shared, d_hidden_routed, d_y_routed,
               d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down)
    references = (*mlp_swiglu_fwd_refs, combine_buffer_ref, output_ref,
                  d_x_ref, mlp_swiglu_bwd_refs[1], mlp_swiglu_bwd_refs[2], mlp_swiglu_bwd_refs[3], mlp_swiglu_bwd_refs[4],
                  mlp_swiglu_bwd_refs[5], mlp_swiglu_bwd_refs[6], mlp_swiglu_bwd_refs[7], d_y_routed_ref, *mlp_swiglu_bwd_refs[8:14])
    difference_stats = []
    for name, out, ref in zip(names, outputs, references):
        if name.endswith("_routed"):
            macrobatch_start = fwd_macrobatch_start if name == "y_routed" else bwd_macrobatch_start
            macrobatch_rows = min(total_routed_tokens - macrobatch_start, MACROBATCH_SIZE)
            macrobatch_valid = valid[macrobatch_start:macrobatch_start + macrobatch_rows]
            out = out[:macrobatch_rows][macrobatch_valid]
            ref = ref[macrobatch_start:macrobatch_start + macrobatch_rows][macrobatch_valid]
        diff = (out.float() - ref.float()).abs()
        diff_sum = diff.sum()
        diff_count = torch.tensor(diff.numel(), dtype=torch.float64, device=device)
        diff_max = diff.max()
        dist.all_reduce(diff_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(diff_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(diff_max, op=dist.ReduceOp.MAX)
        difference_stats.append((name, (diff_sum / diff_count).item(), diff_max.item()))

    fwd_flops = 6 * (NUM_LOCAL_TOKENS + total_routed_tokens) * HIDDEN_DIM * INTERMEDIATE_DIM
    bwd_flops = 2 * fwd_flops
    stats = torch.tensor([total_routed_tokens, fwd_ms, bwd_ms, fwd_flops / 1e9 / fwd_ms, bwd_flops / 1e9 / bwd_ms], dtype=torch.float64, device=device)
    stats_all = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(stats_all, stats)

    if rank == 0:
        print("\nMoE Dispatch + MLP SwiGLU + Combine (forward + backward)")
        print("=========================================================")
        print(f"tokens/rank: {NUM_LOCAL_TOKENS}   experts: {num_experts} ({NUM_LOCAL_EXPERTS}/rank)   "
              f"topk: {TOPK}   H: {HIDDEN_DIM}   I: {INTERMEDIATE_DIM}   macrobatch: {MACROBATCH_SIZE}")
        print(f"iters: warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)
        for name, diff_mean, diff_max in difference_stats:
            print(f"{name:<16} diff mean {diff_mean:.6f}   diff max {diff_max:.6f}")
        print("\nrank  routed tokens  fwd(ms)  bwd(ms)  fwd(TFLOP/s)  bwd(TFLOP/s)")
        print("----  -------------  -------  -------  ------------  ------------")
        for rank_idx, rank_stats in enumerate(stats_all):
            routed_tokens, rank_fwd_ms, rank_bwd_ms, rank_fwd_tflops, rank_bwd_tflops = rank_stats.tolist()
            print(f"{rank_idx:>4}  {int(routed_tokens):>13}  {rank_fwd_ms:>7.3f}  {rank_bwd_ms:>7.3f}  {rank_fwd_tflops:>13.1f}  {rank_bwd_tflops:>13.1f}")

    dist.barrier()
    del x_handle
    del combine_buffer_handle
    del d_combine_buffer_handle
    del d_x_handle
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
