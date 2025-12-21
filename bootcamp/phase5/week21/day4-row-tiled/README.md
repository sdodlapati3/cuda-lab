# Day 4: Row-Tiled GEMM

## Learning Objectives
- Implement row-wise tiling strategy
- Cache a tile of A in shared memory
- Understand partial data reuse
- Measure improvement over naive

## Row Tiling Strategy

### Concept
Instead of each thread independently reading all of A and B,
a thread block cooperatively loads a tile of A into shared memory.

```
Block processes: one tile of rows from A, all columns of B
Each thread: computes one element of output

Tile of A (TILE_SIZE × K) loaded to shared memory
B (K × N) read from global memory
```

### Algorithm
```cpp
__shared__ float As[TILE_SIZE][K];  // Tile of A rows

// 1. Cooperatively load tile of A
for (int k = threadIdx.x; k < K; k += blockDim.x) {
    As[threadIdx.y][k] = A[row * K + k];
}
__syncthreads();

// 2. Each thread computes one output using cached A
float sum = 0;
for (int k = 0; k < K; k++) {
    sum += As[threadIdx.y][k] * B[k * N + col];  // A from shared, B from global
}
```

### Data Reuse Analysis
- **A reads**: M × K → reduced by block width factor
- **B reads**: Still M × N × K (no reuse yet)
- Improvement: Partial, limited by B access

## Why This Helps
- A now read from fast shared memory
- But K × N elements of B still read per block
- Not optimal, but instructive step

## Exercises
1. Implement row-tiled GEMM
2. Compare performance with naive
3. Profile shared memory utilization
4. Explain why B access is still the bottleneck
