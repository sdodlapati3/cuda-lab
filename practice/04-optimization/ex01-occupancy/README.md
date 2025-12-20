# Exercise: Occupancy Analysis

## Objective
Understand and optimize GPU occupancy using the occupancy calculator API.

## Background

Occupancy = active warps / max warps per SM. Higher occupancy can hide memory latency, but isn't always better.

## Limiting Factors
1. **Registers per thread** - Too many registers = fewer concurrent threads
2. **Shared memory per block** - Too much shared memory = fewer concurrent blocks
3. **Block size** - Must be multiple of 32, affects scheduling

## Task

1. Query theoretical occupancy for different kernel configurations
2. Experiment with register limiting (`__launch_bounds__`)
3. Find optimal balance between occupancy and per-thread resources

## CUDA Occupancy API

```cpp
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel, blockSize, dynSharedMem);
cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSize, kernel, 0, 0);
```

## Files

- `occupancy.cu` - Skeleton
- `solution.cu` - Reference
- `Makefile` - Build
- `test.sh` - Test
