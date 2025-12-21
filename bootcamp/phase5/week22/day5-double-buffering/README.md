# Day 5: Double Buffering

## Learning Objectives
- Understand memory latency hiding
- Implement double buffering for GEMM
- Overlap compute with memory loads
- Achieve higher memory bandwidth utilization

## The Problem

In basic tiled GEMM:
```cpp
for each tile:
    Load tile → __syncthreads() → Compute → __syncthreads()
```

The GPU waits for loads to complete before computing.
Memory latency (~400 cycles) is not hidden.

## Double Buffering Solution

Use two sets of shared memory buffers:
```cpp
__shared__ float As[2][TILE][TILE];
__shared__ float Bs[2][TILE][TILE];

Load tile 0 to buffer 0
for t = 1 to numTiles:
    Load tile t to buffer (t % 2)      ← ASYNC
    Compute from buffer ((t-1) % 2)    ← PARALLEL
    __syncthreads()
Compute last tile
```

## Async Memory Operations

CUDA provides `cp.async` for asynchronous global→shared memory copies:

```cpp
// Initiate async copy
__pipeline_memcpy_async(&As[buf][ty][tx], &A[row * K + col], sizeof(float));
__pipeline_commit();

// ... do other work ...

// Wait for copies to complete
__pipeline_wait_prior(0);
```

## Benefits
- Overlaps memory latency with computation
- Higher sustained bandwidth utilization
- Can achieve 10-20% speedup over basic tiling

## Challenges
- More complex code
- More shared memory usage (2× buffers)
- Careful synchronization needed

## Exercises
1. Implement double-buffered GEMM
2. Compare with single-buffered version
3. Profile pipeline utilization
4. Experiment with prefetch distance
