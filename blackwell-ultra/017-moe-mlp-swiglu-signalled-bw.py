"""
To run:
    make
    python3 017-moe-mlp-swiglu-signalled-bw.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import moe_mlp_swiglu


MINIBATCH_SIZE = 4096
NUM_LOCAL_TOKENS = 7168
HIDDEN_DIM = 7168
INTERMEDIATE_DIM = 2048
NUM_LOCAL_EXPERTS = 4
TOPK = 4

WARMUP_ITERS = 5
TIMED_ITERS = 10


def moe_mlp_swiglu_ref(x_shared, x_routed, w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down, tokens_per_expert):
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
    E, H, I = NUM_LOCAL_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM
    routed_capacity = NUM_LOCAL_TOKENS * TOPK * 2
    num_minibatches = (routed_capacity + MINIBATCH_SIZE - 1) // MINIBATCH_SIZE

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Synthetic post-dispatch schedule
    tokens_per_expert_cpu = [NUM_LOCAL_TOKENS * TOPK // E] * E
    tokens_per_expert = torch.tensor(tokens_per_expert_cpu, dtype=torch.int32, device=device)
    total_routed_tokens = sum(tokens_per_expert_cpu)

    # Generate inputs
    gen             = torch.Generator(device=device).manual_seed(1234)
    x_shared        = torch.randn(NUM_LOCAL_TOKENS, H, generator=gen, device=device, dtype=torch.bfloat16)
    x_routed        = torch.randn(routed_capacity, H, generator=gen, device=device, dtype=torch.bfloat16)
    w_shared_gate   = torch.randn(I, H, generator=gen, device=device, dtype=torch.bfloat16) * H ** -0.5
    w_routed_gate   = torch.randn(E, I, H, generator=gen, device=device, dtype=torch.bfloat16) * H ** -0.5
    w_shared_up     = torch.randn(I, H, generator=gen, device=device, dtype=torch.bfloat16) * H ** -0.5
    w_routed_up     = torch.randn(E, I, H, generator=gen, device=device, dtype=torch.bfloat16) * H ** -0.5
    w_shared_down   = torch.randn(H, I, generator=gen, device=device, dtype=torch.bfloat16) * I ** -0.5
    w_routed_down   = torch.randn(E, H, I, generator=gen, device=device, dtype=torch.bfloat16) * I ** -0.5
    dispatch_counter = torch.full((num_minibatches,), 999999, dtype=torch.int32, device=device)
    combine_counter = torch.zeros(num_minibatches, dtype=torch.int32, device=device)
    torch.cuda.synchronize()

    # Benchmark
    for _ in range(WARMUP_ITERS):
        gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed = moe_mlp_swiglu(
            x_shared, x_routed, w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down, tokens_per_expert, dispatch_counter, combine_counter
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed = moe_mlp_swiglu(
            x_shared, x_routed, w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down, tokens_per_expert, dispatch_counter, combine_counter
        )
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / TIMED_ITERS
    flops = 6 * (NUM_LOCAL_TOKENS + total_routed_tokens) * H * I

    print("\nMoE SwiGLU (single-GPU shared + routed expert FFNs, post-dispatch / pre-combine)")
    print("===========================================================================")
    print(f"routed experts: {E}   tokens/expert: {total_routed_tokens // E} ({total_routed_tokens} total)   "
          f"H: {H}   I: {I}   topk: {TOPK}   bf16")
    print(f"shared expert: 1   tokens: {NUM_LOCAL_TOKENS}")
    print(f"routed capacity: {routed_capacity} tokens")
    print(f"compute: {flops / 1e12:.3f} TFLOP")
    print(f"iters:   warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    print("impl         ms      TFLOP/s")
    print("---------  -------  ---------")
    print(f"moe_mlp_swiglu {ms:>7.3f}  {flops / 1e9 / ms:>9.1f}", flush=True)

    # Correctness check
    refs = moe_mlp_swiglu_ref(x_shared, x_routed[:total_routed_tokens], w_shared_gate, w_routed_gate, w_shared_up, w_routed_up, w_shared_down, w_routed_down, tokens_per_expert)
    for name, out, ref in zip(
        ("gate_shared", "gate_routed", "up_shared", "up_routed", "hidden_shared", "hidden_routed", "y_shared", "y_routed"),
        (gate_shared, gate_routed[:total_routed_tokens], up_shared, up_routed[:total_routed_tokens], hidden_shared, hidden_routed[:total_routed_tokens], y_shared, y_routed[:total_routed_tokens]),
        refs
    ):
        diff = (out.float() - ref.float()).abs()
        print(f"\n{name}")
        print(f"out   abs mean {out.abs().mean().item():.4f}   abs max {out.abs().max().item():.4f}")
        print(f"ref   abs mean {ref.abs().mean().item():.4f}   abs max {ref.abs().max().item():.4f}")
        print(f"diff  abs mean {diff.mean().item():.4f}   abs max {diff.max().item():.4f}", flush=True)


if __name__ == "__main__":
    main()
