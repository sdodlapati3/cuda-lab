# Day 4: Tiled Fusion

## Learning Objectives

- Fuse operations in shared memory tiles
- Implement fused tiled matrix operations
- Understand tile size trade-offs

## Key Concepts

### Why Tiled Fusion?

When operations have spatial dependencies (not just element-wise):
- Matrix operations (rows, columns)
- Convolutions (neighborhoods)
- Stencils (neighbors)

Load data once into shared memory, apply multiple operations.

### Tiled Fusion Pattern

```cpp
__global__ void tiled_fused(const float* A, const float* B, float* C) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // Load tile
    tile_A[ty][tx] = A[...];
    tile_B[ty][tx] = B[...];
    __syncthreads();
    
    // Operation 1: use tile
    // Operation 2: use same tile
    // Operation 3: use same tile
    // Only ONE load from global memory!
}
```

### Common Tiled Fusion Patterns

1. **MatMul + Add**: C = A×B + bias
2. **MatMul + Activation**: C = relu(A×B)
3. **BatchNorm + ReLU**: Fuse normalization with activation
4. **Softmax row-wise**: max, exp, sum in one pass per tile

## Build & Run

```bash
./build.sh
./build/tiled_fusion
```
