# Day 1: Tensor Core Architecture

## Learning Objectives
- Understand Tensor Core hardware design
- Learn about matrix fragment sizes
- Understand precision modes
- Compare with CUDA Cores

## What are Tensor Cores?

Tensor Cores are specialized hardware units that perform
matrix multiply-accumulate operations in a single clock cycle.

### Operation
D = A × B + C

Where:
- A: M×K matrix (FP16/BF16/TF32/FP8)
- B: K×N matrix (FP16/BF16/TF32/FP8)
- C, D: M×N matrices (FP16/FP32)

### A100 Tensor Core Specs
- Matrix size: 16×16×16 (WMMA) or 8×8×4 (MMA)
- Peak performance: 312 TFLOPS (FP16)
- 156 TFLOPS (TF32), 19.5 TFLOPS (FP32)

## Precision Modes

| Mode | Input | Accumulate | Use Case |
|------|-------|------------|----------|
| FP16 | FP16  | FP16       | Inference |
| Mixed | FP16 | FP32       | Training |
| TF32 | TF32  | FP32       | Drop-in FP32 |
| BF16 | BF16  | FP32       | Training |
| FP8  | FP8   | FP32       | Inference |

## Comparison: CUDA Core vs Tensor Core

| Metric | CUDA Core | Tensor Core |
|--------|-----------|-------------|
| Operation | Scalar FMA | 16×16×16 MMA |
| Throughput | 1 FMA/cycle | 256 FMA/cycle |
| Peak FP16 | 19.5 TFLOPS | 312 TFLOPS |
| Speedup | 1× | 16× |

## Exercises
1. Calculate theoretical speedup
2. Understand fragment layout
3. Review WMMA documentation
4. Plan first Tensor Core kernel
