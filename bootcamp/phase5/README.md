# Phase 5: GEMM Deep Dive (Weeks 21-28)

## Overview
General Matrix Multiplication (GEMM) is the cornerstone of deep learning compute. This phase provides a comprehensive deep-dive into GEMM optimization, from naive implementations to Tensor Core utilization and CUTLASS integration.

## Why GEMM Matters
- **90%+ of DL compute** is matrix multiplication
- **Convolutions** are implemented as GEMM (im2col)
- **Attention mechanisms** are batched GEMM
- **MLPs** are direct GEMM operations

## Weekly Breakdown

| Week | Topic | Key Concepts |
|------|-------|--------------|
| 21 | GEMM Fundamentals | Naive GEMM, memory access patterns, FLOPs analysis |
| 22 | Tiling Strategies | 2D tiling, shared memory, register blocking |
| 23 | Memory Hierarchy | L2 cache, bank conflicts, double buffering |
| 24 | Vectorized Access | float4, LDG.128, memory coalescing |
| 25 | Warp-Level GEMM | Warp tiles, cooperative loading, warp shuffle |
| 26 | Tensor Cores Intro | WMMA API, FP16 GEMM, accuracy considerations |
| 27 | Advanced Tensor Cores | MMA instructions, pipeline optimization |
| 28 | CUTLASS Integration | Library usage, custom kernels, tuning |

## Performance Targets

| Implementation | % of Peak | TFLOPS (A100) |
|----------------|-----------|---------------|
| Naive | 1-5% | ~3 |
| Tiled (shared memory) | 20-40% | ~60 |
| Register blocking | 50-70% | ~120 |
| Tensor Core naive | 40-60% | ~180 |
| Tensor Core optimized | 80-90% | ~280 |
| cuBLAS/CUTLASS | 90-95% | ~300 |

## Prerequisites
- Completed Phase 1-4
- Understanding of:
  - CUDA memory hierarchy
  - Warp-level programming
  - Shared memory and bank conflicts
  - Basic linear algebra

## The GEMM Operation

```
C = α × A × B + β × C

Where:
- A is M × K matrix
- B is K × N matrix  
- C is M × N matrix
- α, β are scalars
```

### FLOP Count
```
FLOPs = 2 × M × N × K
(one multiply + one add per output element per K)
```

### Memory Bandwidth
```
Bytes = (M×K + K×N + 2×M×N) × sizeof(element)
Arithmetic Intensity = 2×M×N×K / Bytes
```

## Learning Path

```
Week 21: Understand the problem
    ↓
Week 22-23: Master memory hierarchy  
    ↓
Week 24-25: Optimize at warp level
    ↓
Week 26-27: Leverage Tensor Cores
    ↓
Week 28: Production with CUTLASS
```

## Resources
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [NVIDIA GEMM Tutorial](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [How to Optimize GEMM](https://siboehm.com/articles/22/CUDA-MMM)
