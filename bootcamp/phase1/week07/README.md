# Week 7: First Real Kernels

## Overview

Apply everything learned to implement real GPU algorithms from scratch.

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Vector Add & SAXPY | Simplest kernels, bandwidth measurement |
| 2 | Element-wise Operations | Map patterns, function objects |
| 3 | Reduction | Parallel sum, min, max, product |
| 4 | Scan (Prefix Sum) | Inclusive/exclusive scan, applications |
| 5 | Histogram | Atomics, privatization, shared memory |
| 6 | Matrix Multiply | Tiling, shared memory optimization |

## Key Mental Models

### Kernel Categories

| Category | Pattern | Example |
|----------|---------|---------|
| **Map** | 1 input → 1 output | SAXPY, elementwise |
| **Reduce** | N inputs → 1 output | Sum, max, dot product |
| **Scan** | N inputs → N outputs | Prefix sum, cumulative ops |
| **Stencil** | Neighbors → 1 output | Convolution, filters |
| **Scatter** | Write to computed index | Histogram, sort |
| **Gather** | Read from computed index | Sparse matrix ops |

### Performance Metrics

For each kernel, measure:
1. **Bandwidth** (GB/s) - for memory-bound kernels
2. **GFLOPS** - for compute-bound kernels
3. **Efficiency** = Achieved / Theoretical Peak

### Optimization Checklist

- [ ] Coalesced global memory access
- [ ] Shared memory for data reuse
- [ ] No bank conflicts
- [ ] Minimal warp divergence
- [ ] Optimal launch configuration

## Gate Criteria

- [ ] Implement reduction achieving >70% peak bandwidth
- [ ] Implement matrix multiply with shared memory tiling
- [ ] Profile and explain performance characteristics
- [ ] Compare with cuBLAS/CUB library performance

## Reference: CUDA Samples

- `0_Introduction/vectorAdd`
- `6_Performance/reduction`
- `0_Introduction/matrixMul`
