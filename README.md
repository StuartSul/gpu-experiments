# GPU Experiments

A collection of GPU experiments and benchmarks for my personal understanding and research.

## Requirements

- ThunderKittens
- CUDA 12.8+
- NVIDIA Hopper (H100), Blackwell (B200), or Blackwell Ultra (B300) GPUs
- Python 3.11+ with PyTorch 2.8+ and pybind11

## How to Run

1. Run `git submodule update --init --recursive`.
2. In the desired subdirectory, edit the Makefile to target the correct source file, build configuration, and run settings.
3. Run `make run`.

## Organization

- `hopper/`: Experiments targeting H100s
- `blackwell/`: Experiments targeting B200s
- `blackwell-ultra/`: Experiments targetting (G)B300s

## Note

I try my best to keep things organized, but please don’t expect perfect structure / accurate comments / fully working code / etc. Sometimes I make incorrect observations, realize the mistake later, and forget to update the code that led to it.
