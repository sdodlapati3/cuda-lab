# Exercise: Parallel Reduction

## Objective
Implement efficient parallel reduction to compute sum, min, max of large arrays.

## Background

Reduction combines all elements of an array using an associative operator. On GPU, we use tree-based algorithms:

```
Level 0: [a0, a1, a2, a3, a4, a5, a6, a7]
Level 1: [a0+a1, a2+a3, a4+a5, a6+a7]
Level 2: [a0+a1+a2+a3, a4+a5+a6+a7]
Level 3: [a0+a1+a2+a3+a4+a5+a6+a7]
```

## Task

Implement three versions:
1. **Naive reduction** - With warp divergence
2. **Sequential addressing** - No warp divergence  
3. **Warp shuffle reduction** - Using `__shfl_down_sync`

## Key Optimizations

1. **Avoid divergent warps** - Use sequential addressing
2. **Reduce shared memory bank conflicts**
3. **Use warp shuffles** - No shared memory for final warp
4. **Grid-stride loop** - Handle very large arrays

## Files

- `reduction.cu` - Skeleton
- `solution.cu` - Reference
- `Makefile` - Build
- `test.sh` - Test

## Expected Speedup

| Version | Relative Speed |
|---------|----------------|
| Naive | 1x |
| Sequential | ~1.5x |
| Warp shuffle | ~2x |
