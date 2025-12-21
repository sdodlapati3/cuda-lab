# Day 2: 2×2 Thread Tile

## Learning Objectives
- Implement simple 2×2 register blocking
- Understand outer product computation
- Measure improvement over basic tiling
- Analyze register usage

## 2×2 Thread Tile Concept

Each thread computes 4 output elements:
```
Thread output:
┌─────┬─────┐
│ C00 │ C01 │
├─────┼─────┤
│ C10 │ C11 │
└─────┴─────┘
```

## Register Requirements
- 4 accumulators (C00, C01, C10, C11)
- 2 A values (a0, a1)
- 2 B values (b0, b1)
- Total: 8 registers per thread

## Compute Pattern
```cpp
// Load from shared memory
float a0 = As[ty*2][k];
float a1 = As[ty*2+1][k];
float b0 = Bs[k][tx*2];
float b1 = Bs[k][tx*2+1];

// Outer product (4 FMAs)
c00 += a0 * b0;
c01 += a0 * b1;
c10 += a1 * b0;
c11 += a1 * b1;
```

## Efficiency Analysis
- Shared loads per iteration: 4
- FMAs per iteration: 4
- Compute intensity: 1.0 FMA/load (2× over basic)

## Exercises
1. Implement 2×2 thread tile GEMM
2. Compare with basic tiled version
3. Profile register usage
4. Measure speedup
