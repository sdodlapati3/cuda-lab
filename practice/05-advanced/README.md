# Advanced CUDA Exercises

This section contains advanced CUDA exercises covering topics from the latter half of the curriculum.

## Exercises

| Exercise | Topic | Difficulty | Prerequisites |
|----------|-------|------------|---------------|
| ex01-cg-reduction | Cooperative Groups Reduction | ⭐⭐⭐ | Week 6 |
| ex02-cg-scan | CG Parallel Scan | ⭐⭐⭐⭐ | Week 6 |
| ex03-cuda-graphs | CUDA Graphs Optimization | ⭐⭐⭐ | Week 11 |
| ex04-warp-atomics | Warp-Aggregated Atomics | ⭐⭐⭐⭐ | Week 6 |
| ex05-cdp-patterns | Dynamic Parallelism Patterns | ⭐⭐⭐⭐⭐ | Week 15 |
| ex06-cdp-quicksort | CDP Quicksort | ⭐⭐⭐⭐⭐ | Week 15 |
| ex07-cdp-octree | CDP Octree Construction | ⭐⭐⭐⭐⭐ | Week 15 |
| ex08-vmm-growable | VMM Growable Buffer | ⭐⭐⭐⭐⭐ | Week 16 |

## How to Use

1. Navigate to the exercise directory
2. Read the README.md for requirements
3. Implement in the skeleton `.cu` file
4. Test with `./test.sh`
5. Compare with `solution.cu` if stuck

## Compilation Notes

### Cooperative Groups
```bash
nvcc -std=c++17 program.cu -o program
```

### Dynamic Parallelism (CDP)
```bash
nvcc -rdc=true program.cu -o program -lcudadevrt
```

### Virtual Memory Management (VMM)
```bash
nvcc program.cu -o program -lcuda
```

### CUDA Graphs
```bash
nvcc program.cu -o program
```
