# Day 1: Model Loading

## Learning Objectives
- Load model weights efficiently to GPU
- Understand memory layout for weights
- Implement weight sharing patterns

## Key Concepts

### Weight Storage Formats
- Row-major (C-style): A[i][j] at offset i*cols + j
- Column-major (Fortran/BLAS): A[i][j] at offset j*rows + i
- cuBLAS expects column-major by default

### Memory Alignment
- Align weights to 128 bytes for coalescing
- Pad layers to multiples of 8 for tensor cores

### Preloading Strategy
```
1. Calculate total weight memory
2. Allocate one large buffer
3. Load weights in order (layers sequential)
4. Keep pointers to each layer's start
```
