# Week 1: Build System Mastery

> **Goal:** Build, run, and profile CUDA projects with confidence using modern tooling.

## Daily Schedule

| Day | Topic | Deliverable |
|-----|-------|-------------|
| 1 | CMake + Ninja basics | Working CMake project |
| 2 | Compiler flags deep dive | O0 vs O3 benchmark |
| 3 | PTX vs SASS | Annotated SASS analysis |
| 4 | Benchmark harness | Reusable timing framework |
| 5 | Roofline model | GPU roofline plot |
| 6 | Integration | Complete project template |

## Prerequisites

```bash
# Check your environment
nvcc --version          # CUDA toolkit
cmake --version         # CMake 3.18+
ninja --version         # Ninja build system (optional but faster)
ncu --version           # Nsight Compute for profiling
```

## Quick Start

```bash
# Day 1 example
cd day1-cmake-basics
mkdir build && cd build
cmake -G Ninja ..
ninja
./hello_gpu
```

## Week 1 Checklist

- [ ] Can build CUDA projects with CMake
- [ ] Understand `-O3`, `-lineinfo`, `-arch=sm_XX` flags
- [ ] Can extract and read PTX/SASS
- [ ] Have a working benchmark harness
- [ ] Created roofline plot for my GPU
- [ ] Documented learnings in lab notebook

## Resources

- [NVIDIA CMake Guide](https://developer.nvidia.com/blog/building-cuda-applications-cmake/)
- [CUDA Compiler Driver NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [Roofline Model (Berkeley)](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)
