"""
To run:
    make
    python3 013-moe-mlp-swiglu-bw.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import moe_swiglu


NUM_LOCAL_TOKENS = 7168
HIDDEN_DIM = 7168
INTERMEDIATE_DIM = 2048
NUM_LOCAL_EXPERTS = 4
TOPK = 4

WARMUP_ITERS = 5
TIMED_ITERS = 10


def moe_swiglu_ref(x, w_gate, w_up, w_down, tokens_per_expert):
    y = torch.empty_like(x)
    gate = torch.empty(x.size(0), w_gate.size(1), device=x.device, dtype=x.dtype)
    up = torch.empty_like(gate)
    hidden = torch.empty_like(gate)
    offset = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
        _x = x[offset:offset + num_tokens]
        gate[offset:offset + num_tokens] = _x @ w_gate[expert_idx].T
        up[offset:offset + num_tokens] = _x @ w_up[expert_idx].T
        hidden[offset:offset + num_tokens] = (F.silu(gate[offset:offset + num_tokens].float()) * up[offset:offset + num_tokens].float()).to(x.dtype)
        y[offset:offset + num_tokens] = hidden[offset:offset + num_tokens] @ w_down[expert_idx].T
        offset += num_tokens
    return gate, up, hidden, y


def main():
    E, H, I = NUM_LOCAL_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Synthetic post-dispatch schedule
    tokens_per_expert_cpu = [NUM_LOCAL_TOKENS * TOPK // E] * E
    tokens_per_expert = torch.tensor(tokens_per_expert_cpu, dtype=torch.int32, device=device)
    total_tokens = sum(tokens_per_expert_cpu)
    capacity = NUM_LOCAL_TOKENS * TOPK * 2

    # Generate inputs
    gen    = torch.Generator(device=device).manual_seed(1234)
    x      = torch.randn(capacity, H, generator=gen, device=device, dtype=torch.bfloat16)
    w_gate = torch.randn(E, I, H, generator=gen, device=device, dtype=torch.bfloat16) * H ** -0.5
    w_up   = torch.randn(E, I, H, generator=gen, device=device, dtype=torch.bfloat16) * H ** -0.5
    w_down = torch.randn(E, H, I, generator=gen, device=device, dtype=torch.bfloat16) * I ** -0.5
    torch.cuda.synchronize()

    # Benchmark
    for _ in range(WARMUP_ITERS):
        gate, up, hidden, y = moe_swiglu(x, w_gate, w_up, w_down, tokens_per_expert)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        gate, up, hidden, y = moe_swiglu(x, w_gate, w_up, w_down, tokens_per_expert)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / TIMED_ITERS

    flops = 6 * total_tokens * H * I

    print("\nMoE SwiGLU (single-GPU expert FFN, post-dispatch / pre-combine)")
    print("===========================================================================")
    print(f"experts: {E}   tokens/expert: {total_tokens // E} ({total_tokens} total)   "
          f"H: {H}   I: {I}   topk: {TOPK}   bf16")
    print(f"capacity: {capacity} tokens")
    print(f"compute: {flops / 1e12:.3f} TFLOP")
    print(f"iters:   warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    print("impl         ms      TFLOP/s")
    print("---------  -------  ---------")
    print(f"moe_swiglu {ms:>7.3f}  {flops / 1e9 / ms:>9.1f}", flush=True)

    # Correctness check
    refs = moe_swiglu_ref(x[:total_tokens], w_gate, w_up, w_down, tokens_per_expert)
    for name, out, ref in zip(("gate", "up", "hidden", "y"), (gate, up, hidden, y), refs):
        out = out[:total_tokens]
        diff = (out.float() - ref.float()).abs()
        print(f"\n{name}")
        print(f"out   abs mean {out.abs().mean().item():.4f}   abs max {out.abs().max().item():.4f}")
        print(f"ref   abs mean {ref.abs().mean().item():.4f}   abs max {ref.abs().max().item():.4f}")
        print(f"diff  abs mean {diff.mean().item():.4f}   abs max {diff.max().item():.4f}", flush=True)


if __name__ == "__main__":
    main()
