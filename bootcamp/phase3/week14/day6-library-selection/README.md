# Day 6: Library Selection Guide

## Learning Objectives

- Choose the right library for your task
- Understand trade-offs between libraries
- Benchmark and compare approaches

## Decision Framework

### Quick Reference

| Task | First Choice | Alternative |
|------|--------------|-------------|
| Matrix multiply | cuBLAS | CUTLASS |
| Sort large array | Thrust/CUB | Custom radix |
| Prefix scan | CUB | Custom |
| Simple transform | Thrust | Custom kernel |
| FFT | cuFFT | - |
| Sparse matrix | cuSPARSE | - |
| Random numbers | cuRAND | - |
| Deep learning | cuDNN | Custom |

### When to Use What

**cuBLAS** - Dense linear algebra
- Matrix multiply, GEMM
- Tensor core operations
- When math matches BLAS interface

**CUB** - Inside custom kernels
- Block/warp primitives
- Maximum control
- When building custom algorithms

**Thrust** - Rapid development
- STL-like interface
- Automatic memory
- When development time matters

**Custom Kernels** - Specialized needs
- Fused operations
- Unusual patterns
- When libraries don't fit

### Performance Hierarchy

```
cuBLAS/cuDNN (vendor-tuned)
    ↓ usually faster
CUB (optimized primitives)
    ↓ more flexible
Thrust (convenience)
    ↓ easier to write
Custom kernels (full control)
```

## Build & Run

```bash
./build.sh
./build/library_comparison
```
