# Exercise: Warp-Level Reduction

## Objective
Master warp-level primitives (`__shfl_down_sync`, `__shfl_xor_sync`) for ultra-efficient reduction.

## Background

Warp shuffle instructions allow threads within a warp to directly exchange data without shared memory:
- `__shfl_down_sync(mask, val, delta)` - Get value from thread tid+delta
- `__shfl_xor_sync(mask, val, laneMask)` - Butterfly exchange pattern

## Task

1. Implement sum reduction using `__shfl_down_sync`
2. Implement reduction using `__shfl_xor_sync` (butterfly pattern)
3. Implement min/max reduction with warp shuffles

## Files

- `warp_reduction.cu` - Skeleton
- `solution.cu` - Reference
- `Makefile` - Build
- `test.sh` - Test
