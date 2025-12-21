# Week 21: GEMM Fundamentals

## Overview
This week establishes the foundation for GEMM optimization by implementing naive versions and understanding the performance bottlenecks.

## Daily Schedule

| Day | Topic | Focus |
|-----|-------|-------|
| 1 | GEMM Problem Setup | Matrix layout, FLOP counting, roofline model |
| 2 | Naive GEMM | One thread per output element |
| 3 | Memory Access Analysis | Row-major vs column-major, coalescing |
| 4 | Row-Tiled GEMM | Process one row at a time |
| 5 | Column-Tiled GEMM | Process one column at a time |
| 6 | Performance Baseline | Benchmarking, comparison with cuBLAS |

## Key Concepts

### Matrix Layouts
```
Row-major (C-style):    Column-major (Fortran):
[0][1][2]               [0][3][6]
[3][4][5]               [1][4][7]
[6][7][8]               [2][5][8]

Index calculation:
Row-major: A[i][j] = A[i * N + j]
Col-major: A[i][j] = A[i + j * M]
```

### Why GEMM is Hard
1. **Memory-bound for small matrices**: Low arithmetic intensity
2. **Compute-bound for large matrices**: Need high occupancy
3. **Memory access patterns matter**: Non-coalesced = terrible performance

### Performance Analysis
```
Peak FLOPS = SM_count × cores_per_SM × 2 × clock_rate
Peak BW = memory_bandwidth (GB/s)

Achieved FLOPS = 2 × M × N × K / time
Achieved BW = bytes_transferred / time

Arithmetic Intensity = FLOPS / Bytes
```

## Week Goals
- Understand GEMM problem dimensions
- Implement working naive GEMM
- Measure baseline performance
- Identify memory access bottlenecks
- Establish metrics for optimization
