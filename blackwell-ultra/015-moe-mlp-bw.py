"""
To run:
    make
    python3 015-moe-mlp-bw.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import moe_mlp


NUM_LOCAL_TOKENS = 7168
HIDDEN_DIM = 7168
INTERMEDIATE_DIM = 2048
NUM_LOCAL_EXPERTS = 4
TOPK = 4

WARMUP_ITERS = 5
TIMED_ITERS = 10


def main():
    E, H, I = NUM_LOCAL_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Synthetic post-dispatch schedule
    tokens_per_expert = torch.full(
        (E,), NUM_LOCAL_TOKENS * TOPK // E, dtype=torch.int32, device=device
    )
    total_tokens = int(tokens_per_expert.sum().item())

    gen = torch.Generator(device=device).manual_seed(1234)
    a = torch.randn(total_tokens, H, generator=gen, device=device, dtype=torch.bfloat16)
    b = torch.randn(E, I, H, generator=gen, device=device, dtype=torch.bfloat16) * H**-0.5
    d = torch.empty(total_tokens, I, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()

    for _ in range(WARMUP_ITERS):
        moe_mlp(a, b, d, tokens_per_expert)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        moe_mlp(a, b, d, tokens_per_expert)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / TIMED_ITERS

    flops = 2 * total_tokens * H * I

    print("\nMoE MLP grouped up projection")
    print("===========================================================================")
    print(
        f"experts: {E}   tokens/expert: {total_tokens // E} ({total_tokens} total)   "
        f"H: {H}   I: {I}   topk: {TOPK}   bf16"
    )
    print(f"compute: {flops / 1e12:.3f} TFLOP")
    print(f"iters:   warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    print("impl       ms      TFLOP/s")
    print("-------  -------  ---------")
    print(f"moe_mlp  {ms:>7.3f}  {flops / 1e9 / ms:>9.1f}", flush=True)

    d_ref = torch.empty_like(d)
    offset = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
        d_ref[offset : offset + num_tokens] = a[offset : offset + num_tokens] @ b[expert_idx].T
        offset += num_tokens

    out = d.float()
    ref = d_ref.float()
    diff = (out - ref).abs()
    print(f"\nout   abs mean {out.abs().mean().item():.4f}   abs max {out.abs().max().item():.4f}")
    print(f"ref   abs mean {ref.abs().mean().item():.4f}   abs max {ref.abs().max().item():.4f}")
    print(
        f"diff  abs mean {diff.mean().item():.4f}   abs max {diff.max().item():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
