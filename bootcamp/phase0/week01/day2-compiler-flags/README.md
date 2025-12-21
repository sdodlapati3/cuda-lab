# Day 2: Compiler Flags Deep Dive

## Learning Objectives

- Understand optimization levels (`-O0` through `-O3`)
- Know when to use `-lineinfo` vs `-G`
- Master architecture flags (`-arch` vs `-gencode`)
- Pass flags to host compiler with `-Xcompiler`

## The Essential Flags

| Flag | Purpose | When to Use |
|------|---------|-------------|
| `-O0` | No optimization | Debugging only |
| `-O3` | Maximum optimization | Production |
| `-g` | Host debug symbols | Debugging |
| `-G` | Device debug (disables optimizations!) | Debugging only |
| `-lineinfo` | Source mapping without disabling opts | Profiling |
| `-arch=sm_XX` | Target single architecture | Development |
| `-gencode` | Multi-architecture | Distribution |

## Quick Experiments

### 1. Compare O0 vs O3

```bash
./experiments/compare_O0_vs_O3.sh
```

This builds the same kernel with `-O0` and `-O3` and benchmarks both. You'll typically see 5-20x difference!

### 2. Multi-architecture Build

```bash
./experiments/multi_arch_build.sh
```

Builds a single binary that runs on V100, A100, and H100.

### 3. Profile with Line Info

```bash
./experiments/profile_with_lineinfo.sh
```

Shows how Nsight Compute can map instructions back to source.

## Understanding `-arch` vs `-gencode`

### Simple case: Single architecture (development)
```cmake
set(CMAKE_CUDA_ARCHITECTURES 80)  # Just A100
```
Equivalent to: `nvcc -arch=sm_80`

### Production: Multiple architectures
```cmake
set(CMAKE_CUDA_ARCHITECTURES 70 80 90)  # V100, A100, H100
```
Equivalent to:
```bash
nvcc -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_90,code=sm_90
```

### What's the difference?

- `arch=compute_XX` → embed PTX for JIT compilation
- `code=sm_XX` → embed pre-compiled SASS

Best practice for release:
```bash
-gencode arch=compute_90,code=[compute_90,sm_90]
```
This embeds both PTX (for future GPUs) and SASS (for current GPU).

## Common Pitfalls

### 1. Using `-G` in production
```cmake
# NEVER do this for performance!
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
```
`-G` completely disables device optimizations. Your kernel will be 10-100x slower.

### 2. Forgetting `-lineinfo` for profiling
Without it, Nsight shows assembly but can't map to your source code.

### 3. Architecture mismatch
If you build for `sm_90` but run on `sm_80`, you get:
```
CUDA error: no kernel image is available for execution on the device
```

## Exercises

1. Build vector_add with `-O0`, `-O1`, `-O2`, `-O3` and compare performance
2. Use `--ptxas-options=-v` to see register usage at each optimization level
3. Build for `80 90` and verify the binary runs on both architectures
