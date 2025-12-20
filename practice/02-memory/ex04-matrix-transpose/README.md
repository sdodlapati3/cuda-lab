# Exercise: Matrix Transpose Optimization

## Objective
Implement optimized matrix transpose achieving near-peak memory bandwidth.

## Background

Matrix transpose is memory-bound but has a challenge:
- Reading rows = coalesced
- Writing columns = non-coalesced (bad!)

Solution: Use shared memory as intermediate storage.

## Task

1. **Naive transpose** - Direct global memory access
2. **Tiled transpose** - Use shared memory tiles
3. **Conflict-free transpose** - Add padding to eliminate bank conflicts

## Expected Performance

| Version | Bandwidth |
|---------|-----------|
| Naive | ~50 GB/s |
| Tiled | ~200 GB/s |
| Conflict-free | ~300+ GB/s |

## Files

- `matrix_transpose.cu` - Skeleton
- `solution.cu` - Reference
- `Makefile` - Build
- `test.sh` - Test
