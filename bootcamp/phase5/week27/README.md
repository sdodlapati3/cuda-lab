# Week 27: Advanced Tensor Cores

## Overview
This week builds on basic Tensor Core usage to achieve high
performance through tiling, pipelining, and optimal data layouts.

## Daily Schedule

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 1 | Tiled WMMA | Multiple WMMA operations per block |
| 2 | Data Layout Optimization | Fragment-friendly memory layout |
| 3 | Software Pipelining | Overlapping loads and MMA |
| 4 | PTX MMA Instructions | Direct MMA instructions |
| 5 | Multi-Stage Pipeline | Deep pipelining for latency hiding |
| 6 | Performance Tuning | Achieving 80%+ of peak |

## Tiled WMMA GEMM

Instead of one WMMA per warp, tile the problem:

```
Block computes: 128×128 output tile
Warp computes: 64×64 output tile (4×4 WMMA tiles)
Each WMMA: 16×16 output
```

This increases:
- Data reuse (shared memory tiles)
- Compute intensity
- Register utilization

## PTX MMA Instructions

For maximum performance, use PTX mma instructions directly:

```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11, %12, %13};"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3)
);
```

## Expected Performance
After this week: 80-90% of cuBLAS Tensor Core performance
