"""
To run:
    make
    python3 019-mxfp8-quantize.py [M] [N]
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _C import mxfp8_quantize


M = int(sys.argv[1]) if len(sys.argv) > 1 else 28672
N = int(sys.argv[2]) if len(sys.argv) > 2 else 7168
BLOCK_SIZE = 128

WARMUP_ITERS = 5
TIMED_ITERS = 10
GiB = 1024 ** 3

MODES = [
    ("normal",     True,  False),
    ("transposed", False, True),
    ("normal+t",   True,  True),
]


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


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    assert M % BLOCK_SIZE == 0 and N % BLOCK_SIZE == 0

    # Generate inputs and buffers
    gen = torch.Generator(device=device).manual_seed(1234)
    x_bf16 = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    x_bf16.normal_(generator=gen)
    x_fp8 = torch.zeros(M, N, dtype=torch.float8_e4m3fn, device=device)
    x_sc = torch.zeros(M // 128, N // 128, 32, 16, dtype=torch.uint8, device=device)
    x_fp8_t = torch.zeros(N, M, dtype=torch.float8_e4m3fn, device=device)
    x_sc_t = torch.zeros(N // 128, M // 128, 32, 16, dtype=torch.uint8, device=device)

    expected_fp8, expected_sc = mxfp8_quantize_ref(x_bf16)
    expected_fp8_t, expected_sc_t = mxfp8_quantize_ref(x_bf16.t().contiguous())
    expected_fp8 = expected_fp8.view(torch.uint8)
    expected_fp8_t = expected_fp8_t.view(torch.uint8)
    torch.cuda.synchronize()

    print("\nMXFP8 Quantization (normal + transposed)")
    print("===========================================================================")
    print(f"input:   {M} x {N} bf16")
    print(f"normal:  {M} x {N} fp8e4m3 + {M} x {N // 32} e8m0 block scales (swizzled)")
    print(f"transp.: {N} x {M} fp8e4m3 + {N} x {M // 32} e8m0 block scales (swizzled)")
    print(f"iters:   warmup {WARMUP_ITERS}, timed {TIMED_ITERS}\n", flush=True)

    print("mode        correctness  time(ms)  read(GB/s)  write(GB/s)  total(GB/s)")
    print("----------  -----------  --------  ----------  -----------  -----------")
    for name, return_normal, return_transposed in MODES:
        # Correctness check
        x_fp8.zero_(); x_sc.zero_(); x_fp8_t.zero_(); x_sc_t.zero_()
        mxfp8_quantize(x_bf16, x_fp8, x_sc, x_fp8_t, x_sc_t, return_normal, return_transposed)
        errors = 0
        if return_normal:
            errors += int((x_fp8.view(torch.uint8) != expected_fp8).sum().item())
            errors += int((scale_unswizzle(x_sc) != expected_sc).sum().item())
        if return_transposed:
            errors += int((x_fp8_t.view(torch.uint8) != expected_fp8_t).sum().item())
            errors += int((scale_unswizzle(x_sc_t) != expected_sc_t).sum().item())

        # Benchmark
        for _ in range(WARMUP_ITERS):
            mxfp8_quantize(x_bf16, x_fp8, x_sc, x_fp8_t, x_sc_t, return_normal, return_transposed)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMED_ITERS):
            mxfp8_quantize(x_bf16, x_fp8, x_sc, x_fp8_t, x_sc_t, return_normal, return_transposed)
        end.record()
        torch.cuda.synchronize()
        quantize_ms = start.elapsed_time(end) / TIMED_ITERS

        # Performance check
        num_pairs = int(return_normal) + int(return_transposed)
        bytes_read = M * N * 2
        bytes_written = num_pairs * (M * N + M * (N // 32))
        quantize_s = quantize_ms / 1000.0
        ok = "PASSED" if errors == 0 else f"FAILED({errors})"
        print(f"{name:<10}  {ok:<11}  {quantize_ms:>8.3f}  {(bytes_read / GiB) / quantize_s:>10.2f}  "
              f"{(bytes_written / GiB) / quantize_s:>11.2f}  {((bytes_read + bytes_written) / GiB) / quantize_s:>11.2f}", flush=True)


if __name__ == "__main__":
    main()
