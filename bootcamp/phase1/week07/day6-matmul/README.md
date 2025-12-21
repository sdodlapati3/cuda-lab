# Day 6: Matrix Multiply (GEMM)

## Learning Objectives

- Implement the fundamental GPU algorithm
- Master shared memory tiling
- Understand compute vs memory bound transition

## Key Concepts

### Matrix Multiply

C = A × B where:
- A is M × K
- B is K × N  
- C is M × N

Each element: `C[i][j] = sum(A[i][k] * B[k][j]) for k in 0..K`

### Why Matrix Multiply Matters

- Foundation of deep learning (every layer!)
- Most-optimized GPU kernel
- cuBLAS achieves >80% of peak FLOPS
- Understanding it unlocks GPU performance intuition

### Naive Algorithm

```cuda
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[i*K + k] * B[k*N + j];
        }
        C[i*N + j] = sum;
    }
}
```

Problem: Each element of A and B read K times!

### Tiled Algorithm

Load tiles into shared memory, reuse K times:
1. Load TILE_SIZE × TILE_SIZE of A and B into shared memory
2. Compute partial products
3. Move to next tile
4. Repeat until K dimension covered

### Arithmetic Intensity

| Version | Memory Reads | FLOPs | Intensity |
|---------|--------------|-------|-----------|
| Naive | 2MNK | 2MNK | 1 |
| Tiled | 2MNK/TILE | 2MNK | TILE_SIZE |

16×16 tile → 16× data reuse!

## Exercises

1. **Naive GEMM**: Baseline
2. **Tiled GEMM**: Shared memory optimization
3. **Compare cuBLAS**: See the gap

## Build & Run

```bash
./build.sh
./build/matmul
```
