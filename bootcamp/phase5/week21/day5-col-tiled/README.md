# Day 5: Column-Tiled GEMM

## Learning Objectives
- Implement column-wise tiling strategy
- Cache a tile of B in shared memory
- Compare with row-tiled approach
- Understand asymmetry in tiling

## Column Tiling Strategy

### Concept
Instead of caching rows of A, cache columns of B.
Each block processes all rows of A against a tile of B columns.

```
Block processes: all rows of A, one tile of columns from B
Each thread: computes one element of output

A (M × K) read from global memory
Tile of B (K × TILE_SIZE) loaded to shared memory
```

### Algorithm
```cpp
__shared__ float Bs[K][TILE_SIZE];  // Tile of B columns

// 1. Cooperatively load tile of B columns
for (int k = threadIdx.y; k < K; k += blockDim.y) {
    if (col < N) {
        Bs[k][threadIdx.x] = B[k * N + col];
    }
}
__syncthreads();

// 2. Each thread computes one output using cached B
float sum = 0;
for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * Bs[k][threadIdx.x];  // A from global, B from shared
}
```

### Data Reuse Analysis
- **A reads**: Still M × N × K (no reuse)
- **B reads**: K × N → reduced by block height factor
- Similar partial improvement as row-tiling

## Why Tile Both?
Row-tiling: Reduces A reads, B still slow
Column-tiling: Reduces B reads, A still slow
**Solution**: Tile both A and B → 2D tiling (Week 22)

## Exercises
1. Implement column-tiled GEMM
2. Compare with row-tiled
3. Analyze memory access patterns
4. Which is faster and why?
