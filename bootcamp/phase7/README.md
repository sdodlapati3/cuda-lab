# Phase 7: DL Kernels & Attention (Weeks 33-40)

Deep learning kernel optimization and attention mechanisms.

## Overview

| Week | Topic | Focus |
|------|-------|-------|
| 33 | Softmax Optimization | Numerical stability, online algorithms |
| 34 | LayerNorm & RMSNorm | Welford's algorithm, fused kernels |
| 35 | Attention Building Blocks | QK^T, masking, softmax, PV output |
| 36 | Standard MHA Analysis | Memory complexity, roofline analysis |
| 37 | FlashAttention Core | IO-aware tiling, online softmax |
| 38 | FlashAttention Implementation | Forward, backward, FA-2 improvements |
| 39 | Kernel Fusion Strategies | Bias+activation, dropout+residual |
| 40 | Advanced Fusion & Summary | CUTLASS, Triton, phase review |

## Directory Structure

```
phase7/
├── week33/          # Softmax (6 days)
├── week34/          # LayerNorm (6 days)
├── week35/          # Attention Blocks (6 days)
├── week36/          # Standard MHA (6 days)
├── week37/          # FlashAttention Core (6 days)
├── week38/          # FlashAttention Impl (6 days)
├── week39/          # Kernel Fusion (6 days)
└── week40/          # Advanced & Summary (6 days)
```

## Key Algorithms

### Online Softmax (Week 33)
```cpp
// Single-pass softmax
m_new = max(m_old, x_i)
l_new = l_old * exp(m_old - m_new) + exp(x_i - m_new)
y_i = exp(x_i - m_final) / l_final
```

### Welford's Algorithm (Week 34)
```cpp
// Numerically stable variance
count++; delta = x - mean
mean += delta / count
M2 += delta * (x - mean)
variance = M2 / count
```

### FlashAttention Online Update (Week 37-38)
```cpp
// Per-tile rescaling
m_new = max(m_prev, m_tile)
l_new = l_prev * exp(m_prev - m_new) + l_tile * exp(m_tile - m_new)
O_new = (O_prev * l_prev * exp(m_prev - m_new) + P_tile @ V_tile) / l_new
```

## Key Concepts

- **Memory Bound vs Compute Bound**: Most DL ops are memory-bound
- **IO Complexity**: Standard attention O(N²d), FlashAttention O(Nd)
- **Kernel Fusion**: Reduce memory traffic by combining element-wise ops
- **Online Algorithms**: Enable streaming computation without storing intermediates

## Learning Objectives
- ✅ Implement numerically stable softmax
- ✅ Optimize LayerNorm with Welford's algorithm
- ✅ Build attention from primitives (QK^T, mask, softmax, PV)
- ✅ Understand FlashAttention's IO-aware approach
- ✅ Apply kernel fusion strategies

## Prerequisites
- Phase 1-6 completion
- GEMM optimization experience
- cuDNN familiarity
