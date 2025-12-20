# Exercise: Parallel Histogram

## Objective
Implement efficient parallel histogram computation using atomics and privatization.

## Background

Histogram counts occurrences of values. Challenge: many threads updating same bins = contention.

## Optimization Techniques

1. **Global atomics** - Simple but slow due to contention
2. **Shared memory privatization** - Each block has local histogram, merge at end
3. **Warp-level aggregation** - Reduce atomic operations within warp first

## Task

1. Implement naive histogram with global atomics
2. Implement privatized histogram with shared memory
3. Compare performance

## Files

- `histogram.cu` - Skeleton
- `solution.cu` - Reference
- `Makefile` - Build
- `test.sh` - Test

## Expected Speedup

Shared memory privatization typically gives 5-10x speedup over global atomics.
