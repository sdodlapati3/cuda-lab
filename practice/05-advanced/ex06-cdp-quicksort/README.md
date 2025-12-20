# Exercise 06: CDP Quicksort

## Objective
Implement GPU-based quicksort using Dynamic Parallelism.

## Background
Quicksort is naturally recursive - perfect for CDP:
1. Choose pivot
2. Partition array around pivot
3. Recursively sort left and right partitions

With CDP, each partition becomes a child kernel launch!

## Requirements
1. Implement partition step in a kernel
2. Launch child kernels for sub-partitions
3. Switch to insertion sort for small arrays (base case)

## Key Insight
Unlike CPU recursion, GPU CDP can process partitions in parallel across many SMs.

## Testing
```bash
make
./test.sh
```
