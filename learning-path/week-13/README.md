# Week 13: Tensor Cores & Mixed Precision

This week covers modern GPU acceleration using Tensor Cores and mixed precision computing.

## Learning Objectives
- Understand Tensor Core architecture and capabilities
- Implement WMMA (Warp Matrix Multiply-Accumulate)
- Apply mixed precision (FP16/TF32/BF16) for performance
- Leverage cuBLAS with Tensor Core acceleration

## Daily Schedule

| Day | Topic | Notebook |
|-----|-------|----------|
| 1 | Tensor Core Architecture | [day-1-tensor-core-basics.ipynb](day-1-tensor-core-basics.ipynb) |
| 2 | WMMA Programming | [day-2-wmma.ipynb](day-2-wmma.ipynb) |
| 3 | Mixed Precision Training | [day-3-mixed-precision.ipynb](day-3-mixed-precision.ipynb) |
| 4 | cuBLAS Tensor Core Mode | [day-4-cublas-tensor.ipynb](day-4-cublas-tensor.ipynb) |
| 5 | Practice & Quiz | [checkpoint-quiz.md](checkpoint-quiz.md) |

## Prerequisites
- Week 6 (Matrix Operations)
- Week 7 (Memory Optimization)
- Understanding of matrix multiplication

## Hardware Requirements
- NVIDIA GPU with Tensor Cores (Volta V100, Turing T4, Ampere A100, etc.)
- Compute Capability 7.0+

## Key Concepts

### Tensor Core Operations
- 4×4×4 matrix multiply-accumulate per cycle
- Mixed precision: FP16 inputs, FP32 accumulator
- 8-16x speedup over FP32 CUDA cores

### Supported Data Types by Architecture
| Architecture | Supported Types |
|--------------|-----------------|
| Volta (SM70) | FP16 |
| Turing (SM75) | FP16, INT8, INT4 |
| Ampere (SM80) | FP16, BF16, TF32, INT8, INT4 |
| Hopper (SM90) | FP16, BF16, TF32, FP8, INT8 |
