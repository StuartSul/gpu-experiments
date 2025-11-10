# GPU Experiments

A collection of GPU experiments and benchmarks for my personal understanding and research.

## Requirements

- ThunderKittens
- CUDA 12.8+
- NVIDIA Hopper (H100) or Blackwell (B200) GPUs
- Python 3.11+ with PyTorch 2.8+ and pybind11

## How to Run

1. Run `git submodule update --init --recursive`.
2. In the desired subdirectory, edit the Makefile to target the correct source file, build configuration, and run settings.
3. Run `make run`.

## Organization

- `hopper/`: Experiments targeting H100
- `blackwell/`: Experiments targeting B200
