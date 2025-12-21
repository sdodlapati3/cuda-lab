# Day 1: Vector Add & SAXPY

## Learning Objectives

- Implement the simplest useful CUDA kernels
- Measure and analyze memory bandwidth
- Establish performance baselines

## Key Concepts

### Vector Addition

The "Hello World" of GPU computing:
```cuda
C[i] = A[i] + B[i]
```

- Memory bound (3 memory ops, 1 compute op)
- Perfect coalescing opportunity
- Baseline for bandwidth measurement

### SAXPY

Single-precision A*X Plus Y:
```cuda
Y[i] = a * X[i] + Y[i]
```

- Standard BLAS Level 1 operation
- 3 memory ops (2 reads, 1 write), 2 FLOPs
- Memory bandwidth is the limit

### Bandwidth Calculation

```
Bandwidth = Bytes Transferred / Time

For SAXPY: 
- Read X[i]: N * 4 bytes
- Read Y[i]: N * 4 bytes  
- Write Y[i]: N * 4 bytes
- Total: N * 12 bytes

Bandwidth = (N * 12) / time_in_seconds
```

### Comparison to Peak

A100 theoretical peak: ~2 TB/s
- Achieving >80% is excellent
- <50% indicates access pattern issues

## Exercises

1. **Implement vector add**: Measure bandwidth
2. **Implement SAXPY**: Compare to cuBLAS
3. **Vary sizes**: Find saturation point

## Build & Run

```bash
./build.sh
./build/vector_add
./build/saxpy
```
