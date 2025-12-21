# Day 2: Naive GEMM Implementation

## Learning Objectives
- Implement naive GEMM with one thread per output element
- Understand why this approach is inefficient
- Measure baseline performance
- Calculate efficiency vs theoretical peak

## Naive Algorithm

### Pseudocode
```
for each output element C[row][col]:
    sum = 0
    for k = 0 to K-1:
        sum += A[row][k] * B[k][col]
    C[row][col] = sum
```

### CUDA Mapping
```cpp
__global__ void naiveGemm(float* C, float* A, float* B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

## Why Naive is Slow

### Memory Access Pattern
Each thread reads:
- K elements from row `row` of A
- K elements from column `col` of B

For M×N threads:
- Total A reads: M × N × K (massive redundancy!)
- Total B reads: M × N × K (massive redundancy!)

### Actual vs Optimal
- Optimal reads: M×K + K×N elements
- Naive reads: 2 × M × N × K elements
- Redundancy factor: ~2N (for square matrices)

### Memory Coalescing Issues
- A[row * K + k]: Adjacent threads read different rows (not coalesced)
- B[k * N + col]: Adjacent threads read consecutive elements (coalesced)

## Performance Expectation
- Typical: 1-5% of peak TFLOPS
- Limited by memory bandwidth, not compute
- Massive global memory traffic

## Exercises
1. Implement naive GEMM kernel
2. Verify correctness against cuBLAS
3. Measure TFLOPS achieved
4. Calculate efficiency percentage
5. Profile memory throughput with ncu
