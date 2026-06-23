"""
Run on 4 GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 ~/anytitan/.venv/bin/torchrun --nproc_per_node=4 004-naive-comm-comp-overlap.py
"""

import math
import os

import torch
import torch.distributed as dist

from flash_attn.cute.interface import _flash_attn_fwd

# --- experiment parameters ---------------------------------------------------
LOCAL_SEQ = 16384  # per-rank sequence length (global seq = WORLD_SIZE * LOCAL_SEQ)
NHEADS = 128
HEAD_DIM = 128
BATCH = 1
SCALE = 1.0 / math.sqrt(HEAD_DIM)
# -----------------------------------------------------------------------------


def flash_block(q, k, v):
    """Flash attention returning normalized out (B, L, H, D) and lse (B, H, L) fp32."""
    out, lse, _, _ = _flash_attn_fwd(q, k, v, softmax_scale=SCALE, return_lse=True)
    return out, lse


@torch.compile
def combine(acc_o, acc_lse, o, lse):
    """Fuse the online-softmax fold of a partial (o, lse) into the accumulator.

    acc_o (B, L, H, D) fp32 normalized, acc_lse (B, H, L) fp32; o/lse are this step's block.
    """
    new_lse = torch.maximum(acc_lse, lse)
    wa = torch.exp(acc_lse - new_lse)
    wb = torch.exp(lse - new_lse)
    denom = wa + wb
    # (B, H, L) -> (B, L, H, 1) so weights broadcast over head-dim.
    wa = (wa / denom).permute(0, 2, 1).unsqueeze(-1)
    wb = (wb / denom).permute(0, 2, 1).unsqueeze(-1)
    return acc_o * wa + o.float() * wb, new_lse + torch.log(denom)


def ring_exchange(tensors, rank, world_size):
    """Blocking ring shift: send each tensor to rank+1, receive from rank-1."""
    recv = [torch.empty_like(t) for t in tensors]
    ops = [dist.P2POp(dist.isend, t, (rank + 1) % world_size) for t in tensors]
    ops += [dist.P2POp(dist.irecv, r, (rank - 1) % world_size) for r in recv]
    reqs = dist.batch_isend_irecv(ops)
    assert len(reqs) == 1, f"expected 1 coalesced work, got {len(reqs)} (non-NCCL backend?)"
    reqs[0].wait()
    return recv


def ring_attention(q, k, v, rank, world_size):
    """Naive ring attention; q, k, v are this rank's shards (B, L, H, D)."""
    acc_o = acc_lse = None

    for step in range(world_size):
        o, lse = flash_block(q, k, v)

        if acc_o is None:
            acc_o, acc_lse = o.float(), lse
        else:
            acc_o, acc_lse = combine(acc_o, acc_lse, o, lse)

        if step < world_size - 1:
            k, v = ring_exchange([k, v], rank, world_size)

    return acc_o.to(torch.bfloat16)


def reference_full(q, k, v, rank, world_size):
    """All-gather the full sequence, run one flash attention, return this rank's slice."""
    def gather(t):
        full = torch.empty(BATCH, world_size * LOCAL_SEQ, NHEADS, HEAD_DIM, dtype=t.dtype, device=t.device)
        dist.all_gather_into_tensor(full, t.contiguous())
        return full

    full_out, _ = flash_block(gather(q), gather(k), gather(v))
    return full_out[:, rank * LOCAL_SEQ : (rank + 1) * LOCAL_SEQ]


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=local_rank)
    assert dist.get_backend() == "nccl", f"expected NCCL backend, got {dist.get_backend()}"

    if rank == 0:
        print("\nNaive Ring Attention (flash-attn CuTe + NCCL)")
        print("============================================")
        print(f"World size {world_size} | per-rank seq {LOCAL_SEQ} (global {world_size * LOCAL_SEQ})")
        print(f"Heads {NHEADS} | head dim {HEAD_DIM} | batch {BATCH} | {torch.bfloat16}\n", flush=True)
    dist.barrier()

    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    q, k, v = (torch.randn((BATCH, LOCAL_SEQ, NHEADS, HEAD_DIM), dtype=torch.bfloat16, device=device, generator=gen) for _ in range(3))

    out = ring_attention(q, k, v, rank, world_size)
    ref = reference_full(q, k, v, rank, world_size)
    diff = (out.float() - ref.float()).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    ref_abs, out_abs = ref.float().abs(), out.float().abs()
    print(
        f"[rank {rank}] |ring-full| max {max_diff:.5f} mean {mean_diff:.5f} | "
        f"ref max {ref_abs.max().item():.5f} mean {ref_abs.mean().item():.5f} | "
        f"out max {out_abs.max().item():.5f} mean {out_abs.mean().item():.5f}",
        flush=True,
    )
    dist.barrier()

    num_profiles = 3
    num_warmups = 5
    num_iters = 10

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(num_profiles):
            ring_attention(q, k, v, rank, world_size)
        torch.cuda.synchronize()
    trace_path = f"trace_ring_rank{rank}.json"
    prof.export_chrome_trace(trace_path)
    if rank == 0:
        print(f"[rank {rank}] wrote {trace_path} (open in https://ui.perfetto.dev)", flush=True)
    torch.cuda.synchronize()
    dist.barrier()

    for _ in range(num_warmups):
        ring_attention(q, k, v, rank, world_size)
    torch.cuda.synchronize()
    dist.barrier()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        ring_attention(q, k, v, rank, world_size)
    end.record()
    torch.cuda.synchronize()
    dist.barrier()

    avg_ms = start.elapsed_time(end) / num_iters
    flops = 4 * BATCH * NHEADS * HEAD_DIM * LOCAL_SEQ * (world_size * LOCAL_SEQ)
    tflops = flops / (avg_ms * 1e-3) / 1e12
    print(f"[rank {rank}] avg {avg_ms:.3f} ms/iter | {tflops:.1f} TFLOP/s", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
