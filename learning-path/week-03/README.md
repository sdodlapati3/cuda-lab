# Week 3: Parallel Patterns I - Vector Operations

## Overview

This week focuses on implementing fundamental vector operations on the GPU. These form the building blocks for more complex algorithms and help you master grid-stride loops, handling arbitrary data sizes, and fused operations.

## Learning Goals

By the end of this week, you will be able to:
- ✅ Implement grid-stride loops for arbitrary-size data
- ✅ Write efficient element-wise vector operations
- ✅ Understand and implement SAXPY/BLAS-like operations
- ✅ Fuse multiple operations to reduce memory traffic
- ✅ Handle edge cases and boundary conditions

## Daily Schedule

| Day | Topic | Notebook | Key Concepts |
|-----|-------|----------|--------------|
| 1 | Grid-Stride Loops | [day-1-grid-stride-loops.ipynb](day-1-grid-stride-loops.ipynb) | Arbitrary sizes, professional patterns |
| 2 | Element-wise Operations | [day-2-elementwise-ops.ipynb](day-2-elementwise-ops.ipynb) | Add, sub, mul, div, math functions |
| 3 | SAXPY & BLAS Patterns | [day-3-saxpy-blas.ipynb](day-3-saxpy-blas.ipynb) | BLAS Level 1, memory bandwidth |
| 4 | Fused Operations | [day-4-fused-operations.ipynb](day-4-fused-operations.ipynb) | Kernel fusion, multiple ops per element |
| 5 | Review & Quiz | [checkpoint-quiz.md](checkpoint-quiz.md) | Self-assessment |

## Key Concepts

### Grid-Stride Loop Pattern
```python
@cuda.jit
def kernel(data, n):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    for i in range(tid, n, stride):
        # Process element i
        data[i] = process(data[i])
```

### Why Grid-Stride Loops?
1. **Arbitrary sizes**: Handle any N, not just multiples of block size
2. **Flexibility**: Same kernel works for 1K or 1B elements
3. **Occupancy**: Can tune blocks/threads independently of data size
4. **Reusability**: Professional CUDA code pattern

## Week 3 Project: Vector Math Library

Build a complete vector math library with:

```python
class VectorMath:
    def add(a, b, out)       # Element-wise addition
    def sub(a, b, out)       # Element-wise subtraction  
    def mul(a, b, out)       # Element-wise multiplication
    def div(a, b, out)       # Element-wise division
    def scale(a, scalar, out) # Scalar multiplication
    def saxpy(a, x, y, out)  # a*x + y (BLAS operation)
    def dot(a, b) -> float   # Dot product (reduction)
    def norm(a) -> float     # L2 norm (sqrt of sum of squares)
    def fused_mul_add(a, b, c, out)  # a*b + c (FMA)
```

### Deliverables
- [ ] All notebook exercises completed
- [ ] Vector math library implementation
- [ ] Benchmark comparisons vs NumPy
- [ ] Quiz score ≥ 24/30

## Performance Targets

| Operation | Target vs NumPy | Notes |
|-----------|-----------------|-------|
| Element-wise ops | 5-10x faster | For large arrays (>1M elements) |
| SAXPY | 3-5x faster | Memory bandwidth limited |
| Fused operations | 10-20x faster | Reduces memory traffic |

## Resources

- [CUDA C++ Programming Guide - Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
- [BLAS Level 1 Operations](https://netlib.org/blas/#_level_1)
- Week 1-2 notebooks (prerequisites)

---

**Prerequisites:** Complete Week 1 (GPU Fundamentals) and Week 2 (Memory Patterns)

**Next:** Week 4 - Reduction & Atomics
