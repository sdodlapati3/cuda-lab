# Day 5: Roofline Model

## Learning Objectives

- Understand the roofline performance model
- Measure your GPU's peak bandwidth and compute
- Calculate arithmetic intensity
- Position kernels on the roofline chart

## What is the Roofline Model?

The roofline model answers: **"Is my kernel memory-bound or compute-bound?"**

```
                    ┌─────────────────────────────────────
                    │             COMPUTE BOUND
   Performance      │        ╱   (hit the roof)
    (GFLOPS)        │      ╱
                    │    ╱
                    │  ╱  MEMORY BOUND
                    │╱    (on the slope)
                    └─────────────────────────────────────
                         Arithmetic Intensity (FLOP/Byte)
```

**Key insight:** There's a "ridge point" where memory bandwidth meets compute capacity.

## The Math

```
Arithmetic Intensity (AI) = FLOPs / Bytes Accessed

Peak Performance = min(
    Peak Compute (GFLOPS),
    Peak Bandwidth (GB/s) × AI
)
```

### Example: Vector Add
```cpp
c[i] = a[i] + b[i];
// FLOPs: 1 (one addition)
// Bytes: 12 (read 2 floats, write 1 float = 3 × 4 bytes)
// AI = 1/12 ≈ 0.083 FLOP/Byte
```

At AI = 0.083, you're **deeply memory-bound**. Even with 1 TFLOPS of compute, you can only achieve:
```
Performance = 900 GB/s × 0.083 = 75 GFLOPS
```

### Example: GEMM (Matrix Multiply)
```cpp
// For large matrices with good tiling:
// AI ≈ sqrt(N) for NxN matrices
// AI can reach 100+ FLOP/Byte
```

At AI = 100, you're **compute-bound**.

## Quick Start

```bash
mkdir build && cd build
cmake -G Ninja ..
ninja
./roofline_measure   # Measure your GPU's peaks
python3 ../analysis/plot_roofline.py  # Generate plot
```

## Files

```
day5-roofline/
├── CMakeLists.txt
├── src/
│   ├── bandwidth_test.cu    # Measure peak bandwidth
│   ├── compute_test.cu      # Measure peak GFLOPS
│   └── example_kernels.cu   # Position kernels on roofline
├── analysis/
│   ├── plot_roofline.py     # Generate roofline chart
│   └── my_gpu_data.csv      # Your measurements
└── README.md
```

## Key Measurements

| Metric | How to Measure | A100 Example |
|--------|----------------|--------------|
| Peak BW | Memory copy benchmark | ~2 TB/s (HBM2e) |
| Peak FP32 | FFMA loop | ~19.5 TFLOPS |
| Peak FP16 | HFMA loop | ~312 TFLOPS (Tensor) |
| Ridge Point | Peak GFLOPS / Peak GB/s | ~10 FLOP/Byte |

## Common Kernel Positions

| Kernel | AI (FLOP/Byte) | Bound By |
|--------|----------------|----------|
| Vector Add | 0.08 | Memory |
| Reduction | 0.08 | Memory |
| SAXPY | 0.25 | Memory |
| SpMV | 0.125 | Memory |
| Softmax | ~0.5 | Memory |
| LayerNorm | ~0.5 | Memory |
| GEMM (naive) | ~0.5 | Memory |
| GEMM (tiled) | 10-100 | Compute |
| Attention (fused) | 1-10 | Both |

## Exercises

1. Measure your GPU's peak bandwidth and compute
2. Calculate AI for reduction, softmax, and layernorm
3. Create a roofline plot with your kernels positioned
4. Find where the "ridge point" is for your GPU
5. Predict theoretical max performance for each kernel, then verify
