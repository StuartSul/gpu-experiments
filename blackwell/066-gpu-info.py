import torch

props = torch.cuda.get_device_properties(torch.cuda.current_device())

name = props.name
cc = f"{props.major}.{props.minor}"

sm_count = props.multi_processor_count

threads_per_warp = props.warp_size
max_threads_per_sm = props.max_threads_per_multi_processor
num_registers_per_sm = props.regs_per_multiprocessor

default_smem_per_sm = props.shared_memory_per_block
max_smem_per_sm = props.shared_memory_per_multiprocessor
max_smem_per_threadblock = props.shared_memory_per_block_optin

hbm_size = props.total_memory

print("=" * 60)
print(f"GPU Device Properties")
print("=" * 60)
print(f"Device Name:                    {name}")
print(f"Compute Capability:             {cc}")
print(f"Number of SMs:                  {sm_count}")
print()
print(f"Threads per Warp:               {threads_per_warp}")
print(f"Max Threads per SM:             {max_threads_per_sm}")
print(f"Max Warps per SM:               {max_threads_per_sm // threads_per_warp}")
print(f"Max Threadblocks per SM:        {32}") # from NVIDIA Blackwell docs
print()
print(f"Registers per SM:               {num_registers_per_sm:,}")
print()
print(f"Default SMEM per TB:            {default_smem_per_sm // 1024} KB")
print(f"Max SMEM per SM:                {max_smem_per_sm // 1024} KB")
print(f"Max SMEM per Threadblock:       {max_smem_per_threadblock // 1024} KB")
print()
print(f"HBM Size:                       {hbm_size / (1024 ** 3):.1f} GB")
print("=" * 60)
