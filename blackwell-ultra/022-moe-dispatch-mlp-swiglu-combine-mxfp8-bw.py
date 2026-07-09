"""
To run:
    make
    torchrun --nproc_per_node=4 022-moe-dispatch-mlp-swiglu-combine-mxfp8-bw.py
"""

import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import (
    mxfp8_quantize,
    schedule,
    dispatch_mlp_swiglu_combine_fwd,
    dispatch_mlp_swiglu_combine_bwd,
    fwd_epilogue,
    bwd_prologue,
    bwd_epilogue
)


NUM_LOCAL_TOKENS = 7168
HIDDEN_DIM = 7168
INTERMEDIATE_DIM = 2048
NUM_LOCAL_EXPERTS = 4
TOPK = 8
NUM_FWD_COMM_SMS = int(os.environ.get("FWD_COMM_SMS", "40"))
NUM_BWD_COMM_SMS = int(os.environ.get("BWD_COMM_SMS", "40"))
MINIBATCH_SIZE = 16384
MACROBATCH_SIZE = 8 * MINIBATCH_SIZE

WARMUP_ITERS = 5
PROFILE_ITERS = 3
TIMED_ITERS = 10


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

    return x_fp8, x_sc_unswizzled


def scale_unswizzle(sc):
    num_row_blocks, num_col_blocks = sc.shape[0], sc.shape[1]
    sc = sc.view(num_row_blocks, num_col_blocks, 32, 4, 4)
    sc = sc.permute(0, 3, 2, 1, 4)
    return sc.reshape(num_row_blocks * 128, num_col_blocks * 4)


def dequant(x_fp8, sc_unswizzled):
    *E, M, K = x_fp8.shape
    return (
        x_fp8.float().reshape(-1, K // 32, 32) * (2.0 ** (sc_unswizzled.float() - 127.0)).unsqueeze(-1)
    ).view(*E, M, K).to(torch.bfloat16)


def quant_and_dequant(x):
    return dequant(*mxfp8_quantize_ref(x))


def mxfp8_gemm_ref(x, w, out_dtype=torch.bfloat16):
    def to_cublas_scaled_mm_operands(operand):
        fp8, sc = operand if isinstance(operand, tuple) else mxfp8_quantize_ref(operand)
        if sc.dim() == 2:
            M, K32 = sc.shape
            sc = sc.view(M // 128, 4, 32, K32 // 4, 4).permute(0, 3, 2, 1, 4).contiguous()
        return fp8, sc.reshape(-1).view(torch.float8_e8m0fnu)
    x_fp8, x_sc = to_cublas_scaled_mm_operands(x)
    w_fp8, w_sc = to_cublas_scaled_mm_operands(w)
    return torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=x_sc, scale_b=w_sc, out_dtype=out_dtype)


def mlp_swiglu_fwd_ref(
    x_shared, x_routed,
    w_shared_gate, w_routed_gate,
    w_shared_up, w_routed_up,
    w_shared_down, w_routed_down,
    tokens_per_expert
):
    # Shared expert
    gate_shared = x_shared @ w_shared_gate.T
    up_shared = x_shared @ w_shared_up.T
    hidden_shared = ((gate_shared.float() / (torch.exp(-gate_shared.float()) + 1.0)) * up_shared.float()).to(torch.bfloat16)
    y_shared = hidden_shared @ w_shared_down.T

    # Routed expert
    w_routed_gate_fp8, w_routed_gate_sc = w_routed_gate
    w_routed_up_fp8, w_routed_up_sc = w_routed_up
    w_routed_down_fp8, w_routed_down_sc = w_routed_down
    I, H = w_routed_gate_fp8.size(1), w_routed_down_fp8.size(1)

    gate_routed = torch.empty(x_routed.size(0), I, device=x_routed.device, dtype=x_routed.dtype)
    up_routed = torch.empty_like(gate_routed)
    hidden_routed = torch.empty_like(gate_routed)
    y_routed = torch.empty_like(x_routed)

    offset = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
        _x = x_routed[offset:offset + num_tokens]
        gate_routed[offset:offset + num_tokens] = mxfp8_gemm_ref(_x, (w_routed_gate_fp8[expert_idx], w_routed_gate_sc[expert_idx * I // 128:(expert_idx + 1) * I // 128]))
        up_routed[offset:offset + num_tokens] = mxfp8_gemm_ref(_x, (w_routed_up_fp8[expert_idx], w_routed_up_sc[expert_idx * I // 128:(expert_idx + 1) * I // 128]))
        gate_dequant = quant_and_dequant(gate_routed[offset:offset + num_tokens]).float()
        up_dequant = quant_and_dequant(up_routed[offset:offset + num_tokens]).float()
        hidden_routed[offset:offset + num_tokens] = ((gate_dequant / (1.0 + torch.exp(-gate_dequant))) * up_dequant).to(torch.bfloat16)
        y_routed[offset:offset + num_tokens] = mxfp8_gemm_ref(hidden_routed[offset:offset + num_tokens], (w_routed_down_fp8[expert_idx], w_routed_down_sc[expert_idx * H // 128:(expert_idx + 1) * H // 128]))
        offset += num_tokens

    return gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed, y_shared, y_routed


def mlp_swiglu_bwd_ref(
    d_y_shared, d_y_routed,
    x_shared, x_routed,
    gate_shared, gate_routed,
    up_shared, up_routed,
    hidden_shared, hidden_routed,
    w_shared_gate, w_routed_gate_T,
    w_shared_up, w_routed_up_T,
    w_shared_down, w_routed_down_T,
    tokens_per_expert
):
    # Shared expert
    d_hidden_shared = d_y_shared @ w_shared_down
    _denominator = torch.exp(-gate_shared.float()) + 1.0
    _silu = gate_shared.float() / _denominator
    _d_silu = (1.0 - _silu) / _denominator + _silu
    d_gate_shared = ((d_hidden_shared.float() * _d_silu) * up_shared.float()).to(torch.bfloat16)
    d_up_shared = (_silu * d_hidden_shared.float()).to(torch.bfloat16)
    d_x_shared = torch.cat((d_gate_shared, d_up_shared), 1) @ torch.cat((w_shared_gate, w_shared_up), 0)
    d_w_shared_gate = d_gate_shared.T @ x_shared
    d_w_shared_up = d_up_shared.T @ x_shared
    d_w_shared_down = d_y_shared.T @ hidden_shared

    # Routed experts
    num_local_experts = tokens_per_expert.numel()
    num_valid_tokens = int(tokens_per_expert.sum().item())
    w_routed_gate_T_fp8, w_routed_gate_T_sc = w_routed_gate_T
    w_routed_up_T_fp8, w_routed_up_T_sc = w_routed_up_T
    w_routed_down_T_fp8, w_routed_down_T_sc = w_routed_down_T
    H, I = w_routed_gate_T_fp8.size(1), w_routed_down_T_fp8.size(1)
    d_y_routed_fp8, d_y_routed_sc = mxfp8_quantize_ref(d_y_routed)
    gate_routed_dq = quant_and_dequant(gate_routed).float()
    up_routed_dq = quant_and_dequant(up_routed).float()

    d_hidden_routed = torch.zeros_like(gate_routed)
    offset = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
        d_hidden_routed[offset:offset + num_tokens] = mxfp8_gemm_ref(
            (d_y_routed_fp8[offset:offset + num_tokens], d_y_routed_sc[offset:offset + num_tokens]),
            (w_routed_down_T_fp8[expert_idx], w_routed_down_T_sc[expert_idx * I // 128:(expert_idx + 1) * I // 128]))
        offset += num_tokens

    _sigmoid = 1.0 / (1.0 + torch.exp(-gate_routed_dq))
    _silu = gate_routed_dq * _sigmoid
    _d_silu = (1.0 - _silu) * _sigmoid + _silu
    d_gate_routed = ((_d_silu * up_routed_dq) * d_hidden_routed.float()).to(torch.bfloat16)
    d_up_routed = (_silu * d_hidden_routed.float()).to(torch.bfloat16)

    # Dgrad gate+up runs as a single fused GEMM over the concatenated K dimension
    d_gate_fp8, d_gate_sc = mxfp8_quantize_ref(d_gate_routed)
    d_up_fp8, d_up_sc = mxfp8_quantize_ref(d_up_routed)
    d_x_routed = torch.zeros_like(x_routed)
    offset = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
        d_x_routed[offset:offset + num_tokens] = mxfp8_gemm_ref(
            (torch.cat((d_gate_fp8[offset:offset + num_tokens], d_up_fp8[offset:offset + num_tokens]), 1),
             torch.cat((d_gate_sc[offset:offset + num_tokens], d_up_sc[offset:offset + num_tokens]), 1)),
            (torch.cat((w_routed_gate_T_fp8[expert_idx], w_routed_up_T_fp8[expert_idx]), 1),
             torch.cat((w_routed_gate_T_sc[expert_idx * H // 128:(expert_idx + 1) * H // 128],
                        w_routed_up_T_sc[expert_idx * H // 128:(expert_idx + 1) * H // 128]), 1)))
        offset += num_tokens

    # Wgrads consume the transpose-quantized activations (scale blocks run along the token dim)
    d_y_t_fp8, d_y_t_sc = mxfp8_quantize_ref(d_y_routed[:num_valid_tokens].T.contiguous())
    hidden_t_fp8, hidden_t_sc = mxfp8_quantize_ref(hidden_routed[:num_valid_tokens].T.contiguous())
    d_gate_t_fp8, d_gate_t_sc = mxfp8_quantize_ref(d_gate_routed[:num_valid_tokens].T.contiguous())
    d_up_t_fp8, d_up_t_sc = mxfp8_quantize_ref(d_up_routed[:num_valid_tokens].T.contiguous())
    x_t_fp8, x_t_sc = mxfp8_quantize_ref(x_routed[:num_valid_tokens].T.contiguous())

    d_w_routed_gate = torch.empty(num_local_experts, I, H, dtype=torch.bfloat16, device=x_routed.device)
    d_w_routed_up = torch.empty_like(d_w_routed_gate)
    d_w_routed_down = torch.empty(num_local_experts, H, I, dtype=torch.bfloat16, device=x_routed.device)
    offset = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert.tolist()):
        d_w_routed_gate[expert_idx] = mxfp8_gemm_ref(
            (d_gate_t_fp8[:, offset:offset + num_tokens].contiguous(), d_gate_t_sc[:, offset // 32:(offset + num_tokens) // 32].contiguous()),
            (x_t_fp8[:, offset:offset + num_tokens].contiguous(), x_t_sc[:, offset // 32:(offset + num_tokens) // 32].contiguous()))
        d_w_routed_up[expert_idx] = mxfp8_gemm_ref(
            (d_up_t_fp8[:, offset:offset + num_tokens].contiguous(), d_up_t_sc[:, offset // 32:(offset + num_tokens) // 32].contiguous()),
            (x_t_fp8[:, offset:offset + num_tokens].contiguous(), x_t_sc[:, offset // 32:(offset + num_tokens) // 32].contiguous()))
        d_w_routed_down[expert_idx] = mxfp8_gemm_ref(
            (d_y_t_fp8[:, offset:offset + num_tokens].contiguous(), d_y_t_sc[:, offset // 32:(offset + num_tokens) // 32].contiguous()),
            (hidden_t_fp8[:, offset:offset + num_tokens].contiguous(), hidden_t_sc[:, offset // 32:(offset + num_tokens) // 32].contiguous()))
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
    x_buffer = symm_mem.empty(NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    x_buffer.normal_(generator=gen)
    x_buffer_handle = symm_mem.rendezvous(x_buffer, dist.group.WORLD.group_name)
    x_buffer_ptrs = [x_buffer_handle.buffer_ptrs[i] for i in range(world_size)]
    combine_buffer = symm_mem.empty(NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    combine_buffer_handle = symm_mem.rendezvous(combine_buffer, dist.group.WORLD.group_name)
    combine_buffer_ptrs = [combine_buffer_handle.buffer_ptrs[i] for i in range(world_size)]
    d_combine_buffer = symm_mem.empty(NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    d_combine_buffer_handle = symm_mem.rendezvous(d_combine_buffer, dist.group.WORLD.group_name)
    d_combine_buffer_ptrs = [d_combine_buffer_handle.buffer_ptrs[i] for i in range(world_size)]
    d_x_routed_buffer = symm_mem.empty(NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    d_x_routed_buffer_handle = symm_mem.rendezvous(d_x_routed_buffer, dist.group.WORLD.group_name)
    d_x_routed_buffer_ptrs = [d_x_routed_buffer_handle.buffer_ptrs[i] for i in range(world_size)]

    # Generate weights and the output gradient
    w_shared_gate   = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_routed_gate   = torch.randn(NUM_LOCAL_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_shared_up     = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_routed_up     = torch.randn(NUM_LOCAL_EXPERTS, INTERMEDIATE_DIM, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5
    w_shared_down   = torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM, generator=gen, device=device, dtype=torch.bfloat16) * INTERMEDIATE_DIM ** -0.5
    w_routed_down   = torch.randn(NUM_LOCAL_EXPERTS, HIDDEN_DIM, INTERMEDIATE_DIM, generator=gen, device=device, dtype=torch.bfloat16) * INTERMEDIATE_DIM ** -0.5
    d_output        = torch.randn(NUM_LOCAL_TOKENS, HIDDEN_DIM, generator=gen, device=device, dtype=torch.bfloat16) * HIDDEN_DIM ** -0.5

    # MXFP8-quantize the routed weights
    w_routed_gate_fp8, w_routed_gate_sc, _, _ = mxfp8_quantize(w_routed_gate, True, False)
    w_routed_up_fp8, w_routed_up_sc, _, _ = mxfp8_quantize(w_routed_up, True, False)
    w_routed_down_fp8, w_routed_down_sc, _, _ = mxfp8_quantize(w_routed_down, True, False)

    # Pre-transpose the weights for the backward
    w_shared_gate_T = w_shared_gate.transpose(-2, -1).contiguous()
    w_shared_up_T   = w_shared_up.transpose(-2, -1).contiguous()
    w_shared_down_T = w_shared_down.transpose(-2, -1).contiguous()
    w_routed_gate_T_fp8, w_routed_gate_T_sc, _, _ = mxfp8_quantize(w_routed_gate.transpose(-2, -1).contiguous(), True, False)
    w_routed_up_T_fp8, w_routed_up_T_sc, _, _ = mxfp8_quantize(w_routed_up.transpose(-2, -1).contiguous(), True, False)
    w_routed_down_T_fp8, w_routed_down_T_sc, _, _ = mxfp8_quantize(w_routed_down.transpose(-2, -1).contiguous(), True, False)

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
        (x_fp8_t_routed, x_sc_t_routed,
         gate_shared, gate_fp8_routed, gate_sc_routed,
         up_shared, up_fp8_routed, up_sc_routed,
         hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
         y_shared, y_routed) = dispatch_mlp_swiglu_combine_fwd(
            x_buffer, x_buffer_ptrs, combine_buffer, combine_buffer_ptrs,
            w_shared_gate, w_routed_gate_fp8, w_routed_gate_sc,
            w_shared_up, w_routed_up_fp8, w_routed_up_sc,
            w_shared_down, w_routed_down_fp8, w_routed_down_sc,
            schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
            TOPK, NUM_FWD_COMM_SMS, MACROBATCH_SIZE, MINIBATCH_SIZE
        )
        dist.barrier(async_op=True).block_current_stream()
        output = fwd_epilogue(y_shared, combine_buffer, topk_weights)
        return (schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
                x_fp8_t_routed, x_sc_t_routed,
                gate_shared, gate_fp8_routed, gate_sc_routed,
                up_shared, up_fp8_routed, up_sc_routed,
                hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
                y_shared, y_routed, output)

    def run_bwd_once():
        d_y_shared = bwd_prologue(d_output, topk_weights, d_combine_buffer)
        dist.barrier(async_op=True).block_current_stream()
        (d_x_shared, d_x_routed,
         d_gate_shared, d_gate_fp8_routed, d_gate_sc_routed,
         d_up_shared, d_up_fp8_routed, d_up_sc_routed,
         d_hidden_shared, d_hidden_routed, d_y_routed_fp8_routed, d_y_routed_sc_routed,
         d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down) = dispatch_mlp_swiglu_combine_bwd(
            d_y_shared, d_combine_buffer, d_combine_buffer_ptrs, d_x_routed_buffer, d_x_routed_buffer_ptrs,
            w_shared_gate_T, w_routed_gate_T_fp8, w_routed_gate_T_sc,
            w_shared_up_T, w_routed_up_T_fp8, w_routed_up_T_sc,
            w_shared_down_T, w_routed_down_T_fp8, w_routed_down_T_sc,
            x_fp8_t_routed, x_sc_t_routed,
            gate_shared, gate_fp8_routed, gate_sc_routed,
            up_shared, up_fp8_routed, up_sc_routed,
            hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
            x_buffer, x_buffer_ptrs,
            w_routed_gate_fp8, w_routed_gate_sc, w_routed_up_fp8, w_routed_up_sc,
            schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
            TOPK, NUM_BWD_COMM_SMS, MACROBATCH_SIZE, MINIBATCH_SIZE
        )
        dist.barrier(async_op=True).block_current_stream()
        d_x = bwd_epilogue(d_x_shared, d_x_routed_buffer)
        return (d_x, d_x_routed,
                d_gate_shared, d_gate_fp8_routed, d_gate_sc_routed,
                d_up_shared, d_up_fp8_routed, d_up_sc_routed,
                d_hidden_shared, d_hidden_routed, d_y_routed_fp8_routed, d_y_routed_sc_routed,
                d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down)

    # Forward benchmark
    for _ in range(WARMUP_ITERS):
        (schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
         x_fp8_t_routed, x_sc_t_routed,
         gate_shared, gate_fp8_routed, gate_sc_routed,
         up_shared, up_fp8_routed, up_sc_routed,
         hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
         y_shared, y_routed, output) = run_fwd_once()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(TIMED_ITERS):
        run_fwd_once()
    end.record()
    torch.cuda.synchronize()
    dist.barrier()
    fwd_ms = start.elapsed_time(end) / TIMED_ITERS

    # Backward benchmark
    for _ in range(WARMUP_ITERS):
        run_bwd_once()
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

    # A final fwd+bwd for the correctness check
    (schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert,
     x_fp8_t_routed, x_sc_t_routed,
     gate_shared, gate_fp8_routed, gate_sc_routed,
     up_shared, up_fp8_routed, up_sc_routed,
     hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed,
     y_shared, y_routed, output) = run_fwd_once()
    (d_x, d_x_routed,
     d_gate_shared, d_gate_fp8_routed, d_gate_sc_routed,
     d_up_shared, d_up_fp8_routed, d_up_sc_routed,
     d_hidden_shared, d_hidden_routed, d_y_routed_fp8_routed, d_y_routed_sc_routed,
     d_w_shared_gate, d_w_routed_gate, d_w_shared_up, d_w_routed_up, d_w_shared_down, d_w_routed_down) = run_bwd_once()
    torch.cuda.synchronize()
    dist.barrier()

    total_routed_tokens = int(num_tokens.item())

    # Forward reference
    x_all = torch.empty(world_size, NUM_LOCAL_TOKENS, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(x_all, x_buffer)
    valid = schedule_peer_rank >= 0
    x_routed_ref = torch.zeros(schedule_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    x_routed_ref[valid] = x_all[schedule_peer_rank[valid], schedule_peer_token_idx[valid] // TOPK]
    mlp_swiglu_fwd_refs = mlp_swiglu_fwd_ref(
        x_buffer, x_routed_ref,
        w_shared_gate, (w_routed_gate_fp8, w_routed_gate_sc),
        w_shared_up, (w_routed_up_fp8, w_routed_up_sc),
        w_shared_down, (w_routed_down_fp8, w_routed_down_sc),
        tokens_per_expert
    )
    (gate_shared_ref, gate_routed_ref, up_shared_ref, up_routed_ref,
     hidden_shared_ref, hidden_routed_ref, y_shared_ref, y_routed_ref) = mlp_swiglu_fwd_refs
    schedule_peer_rank_all = torch.empty(world_size, schedule_capacity, dtype=torch.int32, device=device)
    schedule_peer_token_idx_all = torch.empty_like(schedule_peer_rank_all)
    y_routed_all = torch.empty(world_size, schedule_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(schedule_peer_rank_all, schedule_peer_rank)
    dist.all_gather_into_tensor(schedule_peer_token_idx_all, schedule_peer_token_idx)
    dist.all_gather_into_tensor(y_routed_all, y_routed_ref)
    combine_buffer_ref = torch.empty_like(combine_buffer)
    for dst_rank in range(world_size):
        dst_valid = schedule_peer_rank_all[dst_rank] == rank
        combine_buffer_ref[schedule_peer_token_idx_all[dst_rank, dst_valid].long()] = y_routed_all[dst_rank, dst_valid]
    output_ref = fwd_epilogue(y_shared_ref, combine_buffer_ref, topk_weights)

    # Backward reference
    d_combine_buffer_ref = torch.empty_like(combine_buffer)
    bwd_prologue(d_output, topk_weights, d_combine_buffer_ref)
    d_combine_buffer_all = torch.empty(world_size, NUM_LOCAL_TOKENS * TOPK, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(d_combine_buffer_all, d_combine_buffer_ref)
    d_y_routed_ref = torch.zeros(schedule_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    d_y_routed_ref[valid] = d_combine_buffer_all[schedule_peer_rank[valid], schedule_peer_token_idx[valid]]
    (d_x_shared_ref, d_x_routed_ref, d_gate_shared_ref, d_gate_routed_ref, d_up_shared_ref, d_up_routed_ref,
     d_hidden_shared_ref, d_hidden_routed_ref,
     d_w_shared_gate_ref, d_w_routed_gate_ref, d_w_shared_up_ref, d_w_routed_up_ref, d_w_shared_down_ref, d_w_routed_down_ref) = mlp_swiglu_bwd_ref(
        d_output, d_y_routed_ref, x_buffer, x_routed_ref,
        gate_shared_ref, gate_routed_ref, up_shared_ref, up_routed_ref, hidden_shared_ref, hidden_routed_ref,
        w_shared_gate, (w_routed_gate_T_fp8, w_routed_gate_T_sc),
        w_shared_up, (w_routed_up_T_fp8, w_routed_up_T_sc),
        w_shared_down, (w_routed_down_T_fp8, w_routed_down_T_sc),
        tokens_per_expert
    )
    d_x_routed_all = torch.empty(world_size, schedule_capacity, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(d_x_routed_all, d_x_routed_ref)
    d_x_routed_buffer_ref = torch.empty_like(d_x_routed_buffer)
    for dst_rank in range(world_size):
        dst_valid = schedule_peer_rank_all[dst_rank] == rank
        d_x_routed_buffer_ref[schedule_peer_token_idx_all[dst_rank, dst_valid].long()] = d_x_routed_all[dst_rank, dst_valid]
    d_x_ref = bwd_epilogue(d_x_shared_ref, d_x_routed_buffer_ref)

    # Dequantize the fused kernel's quantized outputs and build dequantized references
    x_routed_t_dequant = dequant(x_fp8_t_routed, scale_unswizzle(x_sc_t_routed))
    gate_routed_dequant = dequant(gate_fp8_routed, scale_unswizzle(gate_sc_routed))
    up_routed_dequant = dequant(up_fp8_routed, scale_unswizzle(up_sc_routed))
    hidden_routed_t_dequant = dequant(hidden_fp8_t_routed, scale_unswizzle(hidden_sc_t_routed))
    d_y_routed_dequant = dequant(d_y_routed_fp8_routed, scale_unswizzle(d_y_routed_sc_routed))
    d_gate_routed_dequant = dequant(d_gate_fp8_routed, scale_unswizzle(d_gate_sc_routed))
    d_up_routed_dequant = dequant(d_up_fp8_routed, scale_unswizzle(d_up_sc_routed))
    x_routed_t_dequant_ref = dequant(*mxfp8_quantize_ref(x_routed_ref.T.contiguous()))
    gate_routed_dequant_ref = dequant(*mxfp8_quantize_ref(gate_routed_ref))
    up_routed_dequant_ref = dequant(*mxfp8_quantize_ref(up_routed_ref))
    hidden_routed_t_dequant_ref = dequant(*mxfp8_quantize_ref(hidden_routed_ref.T.contiguous()))
    d_y_routed_dequant_ref = dequant(*mxfp8_quantize_ref(d_y_routed_ref))
    d_gate_routed_dequant_ref = dequant(*mxfp8_quantize_ref(d_gate_routed_ref))
    d_up_routed_dequant_ref = dequant(*mxfp8_quantize_ref(d_up_routed_ref))

    # Correctness checks for all returned tensors and final outputs.
    # The forward leaves macrobatch 0 resident; the backward's replay leaves the last macrobatch resident
    num_macrobatches = max(1, (total_routed_tokens + MACROBATCH_SIZE - 1) // MACROBATCH_SIZE)
    fwd_macrobatch_start = 0
    bwd_macrobatch_start = (num_macrobatches - 1) * MACROBATCH_SIZE

    def get_valid_rows(out, ref, start=bwd_macrobatch_start):
        rows = min(total_routed_tokens - start, MACROBATCH_SIZE)
        macrobatch_valid = valid[start:start + rows]
        return out[:rows][macrobatch_valid], ref[start:start + rows][macrobatch_valid]
    def get_valid_cols(out_t, ref_t, start=bwd_macrobatch_start):
        rows = min(total_routed_tokens - start, MACROBATCH_SIZE)
        macrobatch_valid = valid[start:start + rows]
        return out_t[:, :rows][:, macrobatch_valid], ref_t[:, start:start + rows][:, macrobatch_valid]

    difference_stats = []
    for name, out, ref in (
        ("x_routed_t", *get_valid_cols(x_routed_t_dequant, x_routed_t_dequant_ref)),
        ("gate_shared", gate_shared, gate_shared_ref),
        ("gate_routed", *get_valid_rows(gate_routed_dequant, gate_routed_dequant_ref)),
        ("up_shared", up_shared, up_shared_ref),
        ("up_routed", *get_valid_rows(up_routed_dequant, up_routed_dequant_ref)),
        ("hidden_shared", hidden_shared, hidden_shared_ref),
        ("hidden_routed_t", *get_valid_cols(hidden_routed_t_dequant, hidden_routed_t_dequant_ref)),
        ("y_shared", y_shared, y_shared_ref),
        ("y_routed", *get_valid_rows(y_routed, y_routed_ref, fwd_macrobatch_start)),
        ("combine_buffer", combine_buffer, combine_buffer_ref),
        ("output", output, output_ref),
        ("d_y_routed", *get_valid_rows(d_y_routed_dequant, d_y_routed_dequant_ref)),
        ("d_hidden_shared", d_hidden_shared, d_hidden_shared_ref),
        ("d_hidden_routed", *get_valid_rows(d_hidden_routed, d_hidden_routed_ref)),
        ("d_gate_shared", d_gate_shared, d_gate_shared_ref),
        ("d_gate_routed", *get_valid_rows(d_gate_routed_dequant, d_gate_routed_dequant_ref)),
        ("d_up_shared", d_up_shared, d_up_shared_ref),
        ("d_up_routed", *get_valid_rows(d_up_routed_dequant, d_up_routed_dequant_ref)),
        ("d_x_routed", *get_valid_rows(d_x_routed, d_x_routed_ref)),
        ("d_x", d_x, d_x_ref),
        ("d_w_shared_gate", d_w_shared_gate, d_w_shared_gate_ref),
        ("d_w_routed_gate", d_w_routed_gate, d_w_routed_gate_ref),
        ("d_w_shared_up", d_w_shared_up, d_w_shared_up_ref),
        ("d_w_routed_up", d_w_routed_up, d_w_routed_up_ref),
        ("d_w_shared_down", d_w_shared_down, d_w_shared_down_ref),
        ("d_w_routed_down", d_w_routed_down, d_w_routed_down_ref),
    ):
        diff = (out.float() - ref.float()).abs()
        diff_sum = diff.sum()
        diff_count = torch.tensor(diff.numel(), dtype=torch.float64, device=device)
        diff_max = diff.max()
        mismatch = (out.contiguous().view(torch.int16) != ref.contiguous().view(torch.int16)).sum().double()
        dist.all_reduce(diff_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(diff_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(diff_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(mismatch, op=dist.ReduceOp.SUM)
        difference_stats.append((name, (diff_sum / diff_count).item(), diff_max.item(), int(mismatch.item()), int(diff_count.item())))

    fwd_flops = 6 * (NUM_LOCAL_TOKENS + total_routed_tokens) * HIDDEN_DIM * INTERMEDIATE_DIM
    bwd_flops = 2 * fwd_flops
    stats = torch.tensor([total_routed_tokens, fwd_ms, bwd_ms, fwd_flops / 1e9 / fwd_ms, bwd_flops / 1e9 / bwd_ms], dtype=torch.float64, device=device)
    stats_all = [torch.zeros_like(stats) for _ in range(world_size)]
    dist.all_gather(stats_all, stats)

    if rank == 0:
        print("\nMoE Dispatch + MLP SwiGLU + Combine (MXFP8, forward + backward)")
        print("=================================================================")
        print(f"tokens/rank: {NUM_LOCAL_TOKENS}   experts: {num_experts} ({NUM_LOCAL_EXPERTS}/rank)   "
              f"topk: {TOPK}   H: {HIDDEN_DIM}   I: {INTERMEDIATE_DIM}   minibatch: {MINIBATCH_SIZE}   macrobatch: {MACROBATCH_SIZE}")
        print(f"iters: warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)
        for name, diff_mean, diff_max, mismatch, count in difference_stats:
            print(f"{name:<16} diff mean {diff_mean:.6f}   diff max {diff_max:.6f}   mismatched {mismatch}/{count}")
        print("\nrank  routed tokens  fwd(ms)  bwd(ms)  fwd(TFLOP/s)  bwd(TFLOP/s)")
        print("----  -------------  -------  -------  ------------  ------------")
        for rank_idx, rank_stats in enumerate(stats_all):
            routed_tokens, rank_fwd_ms, rank_bwd_ms, rank_fwd_tflops, rank_bwd_tflops = rank_stats.tolist()
            print(f"{rank_idx:>4}  {int(routed_tokens):>13}  {rank_fwd_ms:>7.3f}  {rank_bwd_ms:>7.3f}  {rank_fwd_tflops:>13.1f}  {rank_bwd_tflops:>13.1f}")

    dist.barrier()
    del x_buffer_handle
    del combine_buffer_handle
    del d_combine_buffer_handle
    del d_x_routed_buffer_handle
    gc.collect()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
