# Practice: Memory Patterns

Hands-on exercises for CUDA memory hierarchy and optimization patterns.

## Exercises

| Exercise | Topic | Difficulty | Week |
|----------|-------|------------|------|
| [ex01-coalescing](ex01-coalescing/) | Memory coalescing patterns | ⭐⭐ | Week 2 |
| [ex02-shared-memory](ex02-shared-memory/) | Shared memory tiling | ⭐⭐ | Week 2 |
| [ex03-bank-conflicts](ex03-bank-conflicts/) | Bank conflict avoidance | ⭐⭐⭐ | Week 2 |
| [ex04-matrix-transpose](ex04-matrix-transpose/) | Optimized transpose | ⭐⭐⭐ | Week 6 |

## Learning Objectives

After completing these exercises, you will be able to:

1. **Analyze memory access patterns** - Identify coalesced vs non-coalesced accesses
2. **Use shared memory effectively** - Tile data for reuse and reduced global memory traffic
3. **Avoid bank conflicts** - Understand and prevent shared memory bank conflicts
4. **Optimize data layouts** - Choose between AoS and SoA for GPU efficiency

## Prerequisites

- Week 1 fundamentals (thread indexing, basic kernels)
- Understanding of GPU memory hierarchy

## How to Use

```bash
cd ex01-coalescing
# 1. Read README.md for instructions
# 2. Edit the skeleton .cu file
# 3. Build and test
make
./test.sh
# 4. Compare with solution.cu
```

## Key Concepts

### Memory Coalescing
- Threads in a warp should access consecutive memory addresses
- 128-byte cache line alignment
- Stride-1 access patterns are optimal

### Shared Memory
- 48-96 KB per SM (configurable)
- Much faster than global memory (~100x)
- Used for data reuse within a block

### Bank Conflicts
- 32 banks, 4-byte width
- Threads accessing same bank serialize
- Padding can eliminate conflicts
