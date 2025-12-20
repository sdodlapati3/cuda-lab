# Exercise: CUDA Streams

## Objective
Overlap data transfers with kernel execution using CUDA streams.

## Background

Default (NULL) stream: All operations serialize.
Multiple streams: Operations in different streams can overlap.

## Overlap Requirements
1. Use **pinned (page-locked) memory** for async transfers
2. Use **different streams** for operations that should overlap
3. Device must support concurrent copy and execute

## Task

1. Implement pipeline: H2D → Kernel → D2H using single stream (baseline)
2. Implement overlapped version with multiple streams
3. Measure speedup from overlapping

## Files

- `streams.cu` - Skeleton
- `solution.cu` - Reference
- `Makefile` - Build
- `test.sh` - Test

## Expected Speedup

With proper overlap, expect 1.5-2x speedup for memory-bound workloads.
