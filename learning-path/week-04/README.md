# Week 4: Parallel Patterns II - Reduction & Atomics

## Overview

This week introduces fundamental parallel reduction patterns and atomic operations - essential building blocks for many GPU algorithms including machine learning, statistics, and data processing.

## Learning Objectives

By the end of this week, you will:
- Implement efficient parallel reduction algorithms
- Understand warp-level primitives and shuffle operations
- Use atomic operations for thread-safe updates
- Build histogram and counting kernels

## Daily Schedule

| Day | Topic | Notebook | Key Concepts |
|-----|-------|----------|--------------|
| 1 | Parallel Reduction | [day-1-parallel-reduction.ipynb](day-1-parallel-reduction.ipynb) | Tree reduction, sequential addressing |
| 2 | Warp Primitives | [day-2-warp-primitives.ipynb](day-2-warp-primitives.ipynb) | Warp shuffle, ballot, reduce |
| 3 | Atomic Operations | [day-3-atomic-operations.ipynb](day-3-atomic-operations.ipynb) | atomicAdd, CAS, thread safety |
| 4 | Histogram & Counting | [day-4-histogram.ipynb](day-4-histogram.ipynb) | Binning, privatization, shared atomics |
| 5 | Assessment | [checkpoint-quiz.md](checkpoint-quiz.md) | Self-assessment quiz |

## Prerequisites

- Week 3: Vector operations and grid-stride loops
- Understanding of thread/block hierarchy
- Basic shared memory concepts (Week 2)

## Key Concepts

### Parallel Reduction
```
Input:  [1, 2, 3, 4, 5, 6, 7, 8]
        ↓   ↓   ↓   ↓
Step 1: [3,   7,   11,  15]
        ↓       ↓
Step 2: [10,      26]
        ↓
Step 3: [36]  ← Final sum
```

### Warp Primitives
```
Thread 0  Thread 1  Thread 2  ... Thread 31
   │         │         │            │
   └─────────┴─────────┴────...─────┘
            Warp (32 threads)
         Direct data exchange!
```

### Atomic Operations
```
Thread A ──┐
Thread B ──┼──→ atomicAdd(&counter, 1) ──→ Serialized, safe!
Thread C ──┘
```

## Project: Statistics Library

Build a GPU-accelerated statistics library:

```python
class CUDAStats:
    def sum(self, arr) -> float
    def mean(self, arr) -> float
    def min(self, arr) -> float
    def max(self, arr) -> float
    def histogram(self, arr, bins) -> array
    def variance(self, arr) -> float
```

## Performance Targets

| Operation | CPU Baseline | Week 4 Target |
|-----------|--------------|---------------|
| Sum (10M) | ~10 ms | < 1 ms |
| Min/Max | ~15 ms | < 1 ms |
| Histogram | ~50 ms | < 5 ms |

## Resources

- [NVIDIA Reduction Whitepaper](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- [Warp-Level Primitives Guide](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [Atomic Operations Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#atomic-functions)

## Tips for Success

1. **Understand the reduction tree** - Draw it out!
2. **Minimize atomic contention** - Use privatization
3. **Leverage warp primitives** - Free synchronization within warp
4. **Benchmark everything** - Compare CPU vs GPU
