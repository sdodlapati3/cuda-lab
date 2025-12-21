# Week 26: Tensor Cores Introduction

## Overview
This week introduces NVIDIA Tensor Cores for matrix multiplication.
Tensor Cores provide massive speedup for matrix operations.

## Daily Schedule

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 1 | Tensor Core Architecture | Hardware matrix units |
| 2 | WMMA API | Warp Matrix Multiply-Accumulate |
| 3 | Data Layout | Fragment types and shapes |
| 4 | FP16 GEMM | Half-precision multiplication |
| 5 | Mixed Precision | FP16 compute, FP32 accumulate |
| 6 | Performance Analysis | Tensor Core utilization |

## Tensor Core Basics

### Hardware
- 4th gen Tensor Cores (A100)
- 16×16×16 matrix operations
- FP16 input, FP32 accumulate

### WMMA API
```cpp
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

load_matrix_sync(a_frag, A_ptr, lda);
load_matrix_sync(b_frag, B_ptr, ldb);
fill_fragment(c_frag, 0.0f);

mma_sync(c_frag, a_frag, b_frag, c_frag);

store_matrix_sync(C_ptr, c_frag, ldc, mem_row_major);
```

## Performance Target
Tensor Cores: Up to 312 TFLOPS (FP16)
Target: 40-60% of peak Tensor Core performance
