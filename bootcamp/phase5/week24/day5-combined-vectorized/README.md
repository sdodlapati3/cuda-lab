# Week 24, Day 5: Combined Vectorized GEMM

## Objective
Combine all vectorization techniques into an optimized GEMM kernel.

## Techniques Combined
1. float4 vector loads for global memory
2. Transposed layout for B matrix
3. Async copy for latency hiding
4. Swizzled shared memory for bank conflicts
5. Register blocking (4×4 thread tile)

## Implementation Strategy
- Global → Shared: float4 async copy
- Shared memory: Swizzled layout
- Shared → Register: Scalar loads (already fast)
- Register: Outer product computation

## Expected Results
- 50-60% of cuBLAS
- Significant improvement over individual techniques
