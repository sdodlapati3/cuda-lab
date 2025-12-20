# Exercise: Shared Memory Tiling

## Objective
Implement shared memory tiling to reduce global memory traffic and improve kernel performance.

## Background

Shared memory is on-chip memory (~100x faster than global memory) that can be used to:
1. Cache data that is reused by multiple threads
2. Enable thread cooperation within a block
3. Reduce redundant global memory accesses

### Tiling Pattern
```
1. Load tile from global memory to shared memory
2. __syncthreads()
3. Compute using shared memory
4. __syncthreads() (if needed)
5. Write results to global memory
```

## Task

Implement a 1D stencil operation (e.g., 3-point average) using:
1. **Naive version** - Each thread reads neighbors from global memory
2. **Tiled version** - Load tile to shared memory, compute from shared memory

## Files

- `shared_memory.cu` - Skeleton with TODOs
- `solution.cu` - Reference implementation
- `Makefile` - Build configuration
- `test.sh` - Validation script

## Key Concepts

- `__shared__` keyword for shared memory declaration
- `__syncthreads()` for block synchronization
- Halo/ghost cells for stencil boundaries
- Tile size selection (balance between occupancy and data reuse)

## Expected Speedup

With proper tiling, expect 2-5x speedup for stencil operations due to reduced global memory traffic.
