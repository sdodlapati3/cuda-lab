# Exercise 01: CG Reduction

## Objective
Implement a complete parallel reduction using Cooperative Groups that is more efficient than a naive implementation.

## Requirements
1. Use `cg::reduce()` for warp-level reduction
2. Use shared memory for block-level aggregation
3. Handle arrays of arbitrary size

## Starter Code
Implement the TODO sections in `cg_reduction.cu`.

## Expected Performance
- Should handle 10M elements in < 5ms on modern GPU
- Should give correct sum for all test cases

## Testing
```bash
make
./test.sh
```

## Hints
- Use `cg::thread_block_tile<32>` for warp partitioning
- First reduce within warps, then combine warp results
- Handle partial warps correctly
