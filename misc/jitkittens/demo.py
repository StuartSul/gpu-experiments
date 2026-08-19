from pathlib import Path

import _C as jitkittens
import torch


def main():
    device = torch.device("cuda")
    torch.cuda.init()

    m = n = k = 1024
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    d = torch.empty(m, n, device=device, dtype=torch.bfloat16)

    major, minor = torch.cuda.get_device_capability(device)
    kernel_symbol = "kernel<config<256,64,128,4,true,5,2>>"
    include_directories = [
        str(Path(__file__).with_name("demo.cuh").parents[2] / "ThunderKittens" / "include"),
        *jitkittens.cuda_include_dirs(),
    ]
    cubin, kernel_names = jitkittens.compile_source_to_cubin(
        Path(__file__).with_name("demo.cuh").read_text(),
        [kernel_symbol],
        major, minor,
        include_directories,
        [f"-DKITTENS_SM{major}{minor}", "-DKITTENS_NO_HOST"],
        False,
    )
    module = jitkittens.load_cubin_module(cubin)

    try:
        function = jitkittens.get_kernel_from_cubin_module(module, kernel_names[0])
        data_type = jitkittens.TensorMapDataType.BFLOAT16
        args = jitkittens.create_gl_arguments(
            [data_type, data_type, data_type],
            [a.data_ptr(), b.data_ptr(), d.data_ptr()],
            [[1, 1, m, k], [1, 1, n, k], [1, 1, m, n]],
            [[-1, -1, -1, -1]] * 3,
            [[[128, 128]], [[32, 128]], [[128, 32]]],
            [[128], [128], [64]],
            [[2], [2], [2]],
        )

        dynamic_smem_bytes = 222208
        jitkittens.set_kernel_dynamic_smem(function, dynamic_smem_bytes)
        stream = torch.cuda.current_stream(device)
        jitkittens.launch_kernel(
            function,
            args.data_ptr,
            [128],
            [256],
            dynamic_smem_bytes,
            stream.cuda_stream,
            [2],
            True,
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(d, a @ b.T)
        print("GEMM passed")
    finally:
        jitkittens.unload_cubin_module(module)


if __name__ == "__main__":
    main()
