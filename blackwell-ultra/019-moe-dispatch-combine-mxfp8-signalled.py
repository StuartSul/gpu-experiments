"""
To run:
    make
    torchrun --nproc_per_node=4 019-moe-dispatch-combine-mxfp8-signalled.py
"""

import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import dispatch_combine, schedule


MINIBATCH_SIZE = 4096
NUM_LOCAL_TOKENS = 7168
HIDDEN_DIM = 7168
NUM_LOCAL_EXPERTS = 4
TOPK = 4
BLOCK_SIZE = 128

WARMUP_ITERS = 5
TIMED_ITERS = 10
GiB = 1024 ** 3


@torch.compile
def finalize(combine_recv, topk_weights):
    """
    combine_recv: (NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM) bf16
    topk_weights: (NUM_LOCAL_TOKENS, TOPK) fp32
    returns:      (NUM_LOCAL_TOKENS, HIDDEN_DIM) bf16
    """
    num_local_tokens, topk = topk_weights.shape
    x = combine_recv.view(num_local_tokens, topk, -1).float()
    return (x * topk_weights.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)


def mxfp8_quantize_ref(x):
    M, N = x.shape
    x = x.to(torch.float32)

    # Important: Use explicit float32 constants to match kernel precision
    dest_max = torch.tensor(448.0, dtype=torch.float32, device=x.device)
    min_exp = torch.tensor(-127.0, dtype=torch.float32, device=x.device)
    fp8e8m0_bias = torch.tensor(127.0, dtype=torch.float32, device=x.device)

    block_amax = torch.amax(torch.abs(x).view(M, N // 32, 32), dim=-1)
    decode_scale = torch.clamp(block_amax / dest_max, min=1e-12)
    x_sc_unswizzled = torch.clamp(torch.ceil(torch.log2(decode_scale)), min=min_exp)
    x_fp8 = (x / (2 ** x_sc_unswizzled.repeat_interleave(32, dim=-1))).to(torch.float8_e4m3fn)
    x_sc_unswizzled = (x_sc_unswizzled + fp8e8m0_bias).to(torch.uint8)

    return (
        x_fp8,          # (M, N)
        x_sc_unswizzled # (M, N // 32)
    )


def scale_unswizzle(sc):
    num_row_blocks, num_col_blocks = sc.shape[0], sc.shape[1]
    sc = sc.view(num_row_blocks, num_col_blocks, 32, 4, 4)
    sc = sc.permute(0, 3, 2, 1, 4)
    return sc.reshape(num_row_blocks * 128, num_col_blocks * 4)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

    NUM_EXPERTS = NUM_LOCAL_EXPERTS * world_size
    CAPACITY = NUM_LOCAL_TOKENS * TOPK * 2 # Global recv capacit, 2x headroom over average
    num_minibatches = (CAPACITY + MINIBATCH_SIZE - 1) // MINIBATCH_SIZE

    # Generate inputs and buffers
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    router_logits = torch.randn(NUM_LOCAL_TOKENS, NUM_EXPERTS, generator=gen, device=device)
    schedule_peer_rank = torch.empty((CAPACITY,), dtype=torch.int32, device=device)
    schedule_peer_token_idx = torch.empty((CAPACITY,), dtype=torch.int32, device=device)
    tokens_per_expert = torch.empty(NUM_LOCAL_EXPERTS, dtype=torch.int32, device=device)
    tokens = symm_mem.empty(NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    tokens.normal_(generator=gen)
    dhdl = symm_mem.rendezvous(tokens, dist.group.WORLD.group_name)
    tokens_ptrs = [dhdl.buffer_ptrs[i] for i in range(world_size)]
    dispatch_recv_fp8 = torch.zeros(CAPACITY, HIDDEN_DIM, dtype=torch.float8_e4m3fn, device=device)
    dispatch_recv_sc = torch.zeros(CAPACITY // 128, HIDDEN_DIM // 128, 32, 16, dtype=torch.uint8, device=device)
    dispatch_recv_fp8_t = torch.zeros(HIDDEN_DIM, CAPACITY, dtype=torch.float8_e4m3fn, device=device)
    dispatch_recv_sc_t = torch.zeros(HIDDEN_DIM // 128, CAPACITY // 128, 32, 16, dtype=torch.uint8, device=device)
    dispatch_counter = torch.zeros(num_minibatches, dtype=torch.int32, device=device)
    combine_send = torch.zeros(CAPACITY, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    combine_recv = symm_mem.empty(NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    combine_recv.zero_()
    chdl = symm_mem.rendezvous(combine_recv, dist.group.WORLD.group_name)
    combine_recv_ptrs = [chdl.buffer_ptrs[i] for i in range(world_size)]
    combine_counter = torch.full((num_minibatches,), 999999, dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print("\nMoE Dispatch + Combine Bandwidth")
        print("===========================================================================")
        print(f"tokens/rank: {NUM_LOCAL_TOKENS} x {HIDDEN_DIM} bf16   experts: {NUM_EXPERTS} "
              f"({NUM_LOCAL_EXPERTS}/rank)   topk: {TOPK}")
        print(f"recv cap:    {CAPACITY} rows total")
        print(f"iters:       warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    # Router all-gather
    topk_vals, topk_ids = torch.topk(router_logits, TOPK, dim=1)
    topk_weights = torch.softmax(topk_vals.float(), dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    topk_ids_all = torch.empty(world_size, NUM_LOCAL_TOKENS, TOPK, dtype=torch.int32, device=device)
    dist.all_gather_into_tensor(topk_ids_all, topk_ids)

    # Scheduler benchmark
    for _ in range(WARMUP_ITERS):
        schedule(topk_ids_all, schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert, rank)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        schedule(topk_ids_all, schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert, rank)
    end.record()
    torch.cuda.synchronize()
    scheduler_ms = start.elapsed_time(end) / TIMED_ITERS

    # Fill in combine_send buffer
    tokens_all = torch.empty(world_size, NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(tokens_all, tokens)
    valid = schedule_peer_rank >= 0
    combine_send[valid] = tokens_all[schedule_peer_rank[valid], schedule_peer_token_idx[valid] // TOPK]

    # Benchmark
    for _ in range(WARMUP_ITERS):
        dispatch_combine(
            tokens, tokens_ptrs,
            dispatch_recv_fp8, dispatch_recv_sc, dispatch_recv_fp8_t, dispatch_recv_sc_t,
            combine_send, combine_recv, combine_recv_ptrs,
            schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert,
            dispatch_counter, combine_counter, TOPK
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()  # release all ranks together so communication is genuinely concurrent
    start.record()
    for _ in range(TIMED_ITERS):
        dispatch_combine(
            tokens, tokens_ptrs,
            dispatch_recv_fp8, dispatch_recv_sc, dispatch_recv_fp8_t, dispatch_recv_sc_t,
            combine_send, combine_recv, combine_recv_ptrs,
            schedule_peer_rank, schedule_peer_token_idx, tokens_per_expert,
            dispatch_counter, combine_counter, TOPK
        )
    end.record()
    torch.cuda.synchronize()
    dispatch_combine_ms = start.elapsed_time(end) / TIMED_ITERS
    dist.barrier()

    # Finalize benchmark
    for _ in range(WARMUP_ITERS):
        output = finalize(combine_recv, topk_weights)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        output = finalize(combine_recv, topk_weights)
    end.record()
    torch.cuda.synchronize()
    finalize_ms = start.elapsed_time(end) / TIMED_ITERS

    # Dispatch correctness check
    dispatch_expected = tokens_all[schedule_peer_rank[valid], schedule_peer_token_idx[valid] // TOPK]

    expected_fp8, expected_sc = mxfp8_quantize_ref(dispatch_expected)
    actual_fp8 = dispatch_recv_fp8.view(torch.uint8)
    actual_sc = scale_unswizzle(dispatch_recv_sc)
    dispatch_errors = (actual_fp8[valid] != expected_fp8.view(torch.uint8)).sum()
    dispatch_errors += (actual_sc[valid] != expected_sc).sum()

    expected_fp8_t, expected_sc_t = mxfp8_quantize_ref(combine_send.t().contiguous())  # hack
    actual_fp8_t = dispatch_recv_fp8_t.view(torch.uint8)
    actual_sc_t = scale_unswizzle(dispatch_recv_sc_t)
    is_processed = torch.arange(CAPACITY // 32, device=device) * 32 < tokens_per_expert.sum()
    expected_sc_t = torch.where(is_processed[None, :], expected_sc_t, torch.zeros_like(expected_sc_t))
    dispatch_errors += (actual_fp8_t != expected_fp8_t.view(torch.uint8)).sum()
    dispatch_errors += (actual_sc_t != expected_sc_t).sum()
    dist.all_reduce(dispatch_errors, op=dist.ReduceOp.SUM)

    # Combine correctness check
    combine_expected = tokens.unsqueeze(1).expand(NUM_LOCAL_TOKENS, TOPK, HIDDEN_DIM).reshape(-1, HIDDEN_DIM)
    combine_errors = (combine_recv != combine_expected).sum()
    dist.all_reduce(combine_errors, op=dist.ReduceOp.SUM)

    # Finalize correctness check
    finalize_errors = ((output.float() - tokens.float()).abs() > 1e-2).sum()
    dist.all_reduce(finalize_errors, op=dist.ReduceOp.SUM)

    # Performance check
    num_tokens = int(valid.sum().item())
    num_local_tokens = int((valid & (schedule_peer_rank == rank)).sum().item())
    num_remote_tokens = num_tokens - num_local_tokens
    bytes_total = num_tokens * HIDDEN_DIM * 2
    bytes_remote = num_remote_tokens * HIDDEN_DIM * 2
    bytes_finalize = (NUM_LOCAL_TOKENS * TOPK + NUM_LOCAL_TOKENS) * HIDDEN_DIM * 2
    stats = torch.tensor([
        num_tokens, 
        num_local_tokens, num_remote_tokens, 
        bytes_total, bytes_remote, bytes_finalize,
        dispatch_combine_ms, scheduler_ms, finalize_ms
    ], dtype=torch.float64, device=device)
    stats_all = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(stats_all, stats)

    if rank == 0:
        dispatch_ok = int(dispatch_errors.item()) == 0
        combine_ok = int(combine_errors.item()) == 0
        finalize_ok = int(finalize_errors.item()) == 0
        print(f"dispatch correctness: {'PASSED' if dispatch_ok else 'FAILED (' + str(int(dispatch_errors.item())) + ' mismatches)'}")
        print(f"combine  correctness: {'PASSED' if combine_ok else 'FAILED (' + str(int(combine_errors.item())) + ' mismatches)'}")
        print(f"finalize correctness: {'PASSED' if finalize_ok else 'FAILED (' + str(int(finalize_errors.item())) + ' mismatches)'}\n")
        print("rank  tokens  local  remote  comm(ms)  total(GB/s)  remote(GB/s)  sched(ms)  fin(ms)  fin(GB/s)")
        print("----  ------  -----  ------  --------  -----------  ------------  ---------  -------  ---------")
        max_comm_s = 0.0
        sum_remote_bytes = 0.0
        sum_total_bytes = 0.0
        for rank, stats in enumerate(stats_all):
            nt, lt, rt, bt, br, bf, dcms, sms, fms = stats.tolist()
            comm_s = dcms / 1000.0
            fin_s = fms / 1000.0
            print(f"{int(rank):>4}  {int(nt):>6}  {int(lt):>5}  {int(rt):>6}  "
                  f"{dcms:>8.3f}  {(2 * bt / GiB) / comm_s:>11.2f}  {(2 * br / GiB) / comm_s:>12.2f}  "
                  f"{sms:>9.4f}  {fms:>7.3f}  {(bf / GiB) / fin_s:>9.2f}")
            max_comm_s = max(max_comm_s, comm_s)
            sum_remote_bytes += br
            sum_total_bytes += bt
        print("----  ------  -----  ------  --------  -----------  ------------  ---------  -------  ---------")
        print(f"dispatch + combine (sum bytes / slowest rank): "
              f"total {(2 * sum_total_bytes / GiB) / max_comm_s:.2f} GB/s, "
              f"remote {(2 * sum_remote_bytes / GiB) / max_comm_s:.2f} GB/s", flush=True)

    dist.barrier()
    del dhdl
    del chdl
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
