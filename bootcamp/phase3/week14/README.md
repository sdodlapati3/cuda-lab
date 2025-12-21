# Week 14: CUDA Libraries

## Theme: Standing on the Shoulders of Giants

Why reinvent the wheel? This week covers production-quality CUDA libraries that you should use instead of writing custom kernels.

## Daily Breakdown

| Day | Topic | Focus |
|-----|-------|-------|
| 1 | cuBLAS Basics | GEMM, vector operations |
| 2 | cuBLAS Advanced | Batched ops, tensor cores |
| 3 | CUB Primitives | Device-wide algorithms |
| 4 | CUB Advanced | Block/warp-level, custom ops |
| 5 | Thrust | High-level STL-like algorithms |
| 6 | Library Selection | When to use which |

## Key Libraries

| Library | Purpose | When to Use |
|---------|---------|-------------|
| cuBLAS | Linear algebra | Matrix multiply, BLAS ops |
| cuDNN | Neural networks | Convolutions, activations |
| CUB | Parallel primitives | Scan, reduce, sort, select |
| Thrust | High-level algorithms | Quick prototyping, transforms |
| cuSPARSE | Sparse matrices | Sparse linear algebra |
| cuFFT | FFT operations | Signal processing |
| cuRAND | Random numbers | Monte Carlo, initialization |

## Mental Model: Build vs Use

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION TREE                            │
├─────────────────────────────────────────────────────────────┤
│ Is there a library for this?                                │
│   ├── Yes → USE THE LIBRARY                                 │
│   │         (cuBLAS, CUB, Thrust, etc.)                     │
│   │                                                         │
│   └── No → Do you need fusion?                              │
│            ├── Yes → Write custom kernel                    │
│            └── No → Can you compose library calls?          │
│                     ├── Yes → Use composition               │
│                     └── No → Write custom kernel            │
└─────────────────────────────────────────────────────────────┘
```

## This Week's Goal

By the end of Week 14, you can:
- Use cuBLAS for matrix operations
- Apply CUB for device-wide algorithms
- Leverage Thrust for rapid development
- Know when to use libraries vs custom code
