# Day 1: cuBLAS Basics

## Learning Objectives

- Set up and use cuBLAS
- Perform vector and matrix operations
- Understand row-major vs column-major

## Key Concepts

### What is cuBLAS?

cuBLAS is NVIDIA's GPU-accelerated implementation of BLAS (Basic Linear Algebra Subprograms).

- Level 1: Vector operations (axpy, dot, nrm2)
- Level 2: Matrix-vector operations (gemv)
- Level 3: Matrix-matrix operations (gemm)

### Basic Usage Pattern

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

// ... do operations ...

cublasDestroy(handle);
```

### Column-Major Order (IMPORTANT!)

cuBLAS uses **column-major** order (like Fortran), not row-major (like C).

```
Row-major (C/C++):       Column-major (cuBLAS):
[0 1 2]                  [0 3 6]
[3 4 5]     stored as    [1 4 7]     stored as
[6 7 8]     [0,1,2,3,4,  [2 5 8]     [0,1,2,3,4,
             5,6,7,8]                  5,6,7,8]
```

### GEMM: General Matrix Multiply

```cpp
// C = α * A * B + β * C
cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
    m, n, k,                    // Dimensions
    &alpha,                     // Scalar α
    d_A, lda,                   // Matrix A, leading dimension
    d_B, ldb,                   // Matrix B, leading dimension
    &beta,                      // Scalar β
    d_C, ldc);                  // Matrix C, leading dimension
```

## Build & Run

```bash
./build.sh
./build/cublas_basics
```
