# Day 4: Roofline Analysis

## What You'll Learn

- Understand the roofline model
- Calculate arithmetic intensity
- Identify kernel bounds (memory vs compute)
- Use roofline for optimization decisions

## The Roofline Model

```
                    Compute Ceiling
Performance         ___________________
(FLOP/s)           /
                  /
                 /
                /  <-- Memory Bandwidth Ceiling
               /
              /
             /
            -------------------------
              Arithmetic Intensity (FLOP/Byte)
```

### Key Concepts

**Arithmetic Intensity (AI)**:
```
AI = FLOPs / Bytes Transferred
```

**Ridge Point**:
```
Ridge Point AI = Peak FLOP/s / Peak Bandwidth
```
- Below ridge: Memory-bound
- Above ridge: Compute-bound

## A100 Example

```
Peak FP32: ~19.5 TFLOP/s
Peak Bandwidth: ~2 TB/s
Ridge Point: 19.5 / 2 ≈ 10 FLOP/Byte

If your kernel has:
  AI = 0.5  → Memory-bound (most kernels)
  AI = 50   → Compute-bound
```

## Calculating AI for Common Operations

### Vector Add
```cpp
c[i] = a[i] + b[i];
FLOPs: 1
Bytes: 3 × 4 = 12 (2 reads + 1 write)
AI = 1/12 ≈ 0.08  → Very memory-bound
```

### Matrix Multiply (naive)
```cpp
C[i][j] += A[i][k] * B[k][j];
FLOPs: 2N³
Bytes: 3N² × 4
AI = 2N³ / (12N²) = N/6  → Scales with N
```

### SAXPY
```cpp
y[i] = a * x[i] + y[i];
FLOPs: 2 (multiply + add)
Bytes: 3 × 4 = 12
AI = 2/12 ≈ 0.17  → Memory-bound
```

## Quick Start

```bash
./build.sh

# Run roofline analysis
./build/roofline_demo

# Profile with ncu roofline
ncu --set roofline -o roofline_report ./build/roofline_kernels
```

## ncu Roofline Section

```bash
ncu --set roofline ./my_app
```

Shows:
- Kernel position on roofline chart
- Achieved vs peak performance
- Memory vs compute bound indicator

## Exercises

1. Calculate AI for your kernels
2. Plot kernels on roofline chart
3. Identify optimization strategy
4. Improve AI through tiling/blocking
