# Day 2: cuBLAS Advanced

## Learning Objectives

- Use batched operations
- Leverage Tensor Cores
- Understand mixed precision

## Key Concepts

### Batched Operations

Process many small matrices simultaneously:

```cpp
// Batched GEMM: C[i] = A[i] * B[i] for i = 0..batch-1
cublasSgemmBatched(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    d_A_array, lda,    // Array of pointers to A matrices
    d_B_array, ldb,    // Array of pointers to B matrices
    &beta,
    d_C_array, ldc,    // Array of pointers to C matrices
    batch_count);
```

### Strided Batched (More Efficient)

For contiguous batches:

```cpp
// Matrices stored contiguously with stride
cublasSgemmStridedBatched(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    d_A, lda, stride_A,  // A matrices with stride between them
    d_B, ldb, stride_B,
    &beta,
    d_C, ldc, stride_C,
    batch_count);
```

### Tensor Cores (NVIDIA Ampere+)

For massive throughput on supported types:

```cpp
// Enable Tensor Core math
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// Use cublasGemmEx for mixed precision
cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    d_A, CUDA_R_16F, lda,  // FP16 input
    d_B, CUDA_R_16F, ldb,
    &beta,
    d_C, CUDA_R_32F, ldc,  // FP32 output
    CUDA_R_32F,            // Compute type
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

## Build & Run

```bash
./build.sh
./build/cublas_advanced
```
