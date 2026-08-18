import ctypes
from pathlib import Path

import _C as jitkittens
import torch


def main():
    device = torch.device("cuda")
    n = 1024
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)
    c = torch.empty_like(a)

    major, minor = torch.cuda.get_device_capability(device)
    source = Path(__file__).with_name("demo.cuh").read_text()
    cubin, kernel_names = jitkittens.compile_source_to_cubin(source, ["add"], major, minor)
    module = jitkittens.load_cubin_module(cubin)

    try:
        function = jitkittens.get_kernel_from_cubin_module(module, kernel_names[0])
        arg_a = ctypes.c_void_p(a.data_ptr())
        arg_b = ctypes.c_void_p(b.data_ptr())
        arg_c = ctypes.c_void_p(c.data_ptr())
        arg_n = ctypes.c_int(n)
        args = (ctypes.c_void_p * 4)(
            ctypes.addressof(arg_a), ctypes.addressof(arg_b), ctypes.addressof(arg_c), ctypes.addressof(arg_n)
        )

        block_size = 256
        jitkittens.launch_kernel(
            function,
            ctypes.addressof(args),
            [(n + block_size - 1) // block_size],
            [block_size],
            stream=torch.cuda.current_stream(device).cuda_stream,
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(c, a + b)
        print("add kernel passed")
    finally:
        jitkittens.unload_cubin_module(module)


if __name__ == "__main__":
    main()
