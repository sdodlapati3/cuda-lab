# Day 2: Basic Tiled GEMM

## Learning Objectives
- Implement 2D tiled GEMM kernel
- Understand synchronization requirements
- Compare performance with naive implementation
- Verify correctness against cuBLAS

## Algorithm

```cpp
template<int TILE_SIZE>
__global__ void tiledGemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Iterate over tiles along K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative load of tiles
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        
        __syncthreads();  // Wait for all loads
        
        // Compute partial dot product from shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();  // Wait before loading next tile
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Key Points

### Synchronization
Two `__syncthreads()` are essential:
1. After loading: Ensure all threads have loaded before compute
2. After compute: Ensure all threads are done before loading next tile

### Boundary Handling
- Load 0.0f for out-of-bounds elements
- This handles non-multiple tile sizes correctly

### Memory Access Pattern
- Global loads are coalesced (consecutive threads access consecutive addresses)
- Shared memory accesses avoid bank conflicts with this layout

## Performance Expectation
With TILE_SIZE=32:
- ~32× reduction in global memory traffic
- Expected: 20-40% of cuBLAS performance
- Significant improvement over naive (~5-10× faster)

## Exercises
1. Implement the tiled GEMM kernel
2. Verify correctness against cuBLAS
3. Profile memory bandwidth utilization
4. Experiment with boundary conditions
