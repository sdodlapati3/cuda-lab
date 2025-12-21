# Week 24, Day 2: Transposed Loads

## Objective
Optimize memory access patterns by loading transposed data for better coalescing.

## Key Concepts
- Row-major vs column-major access patterns
- Transposing during load to shared memory
- Coalesced global memory reads
- Layout transformation for GEMM

## Why Transpose?
- Matrix A accessed row-wise
- Matrix B accessed column-wise (for C = A × B)
- Column access is strided → poor coalescing
- Transpose B so reads become row-wise

## Implementation
- Load B^T instead of B
- Store transposed in shared memory
- Compute proceeds as normal

## Expected Results
- Better memory coalescing
- 10-20% improvement for column-accessed matrices
