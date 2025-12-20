# Exercise 03: CUDA Graphs Optimization

## Objective
Convert a multi-kernel workflow to use CUDA Graphs for reduced launch overhead.

## Requirements
1. Capture a kernel sequence into a CUDA Graph
2. Create and launch a graph executable
3. Compare performance vs. individual kernel launches

## Background
CUDA Graphs provide:
- Reduced CPU-side kernel launch overhead
- Optimized scheduling of kernel sequences
- Ability to replay the same work efficiently

## Workflow to Optimize
1. Initialize array (kernel 1)
2. Square all elements (kernel 2)
3. Add constant (kernel 3)
4. Reduce sum (kernel 4)

## Testing
```bash
make
./test.sh
```

## Expected Outcome
Graph execution should be faster than individual launches, especially for small kernels.
