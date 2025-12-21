# Week 24, Day 1: float4 Vector Loads

## Objective
Introduction to vectorized memory access using float4 for GEMM optimization.

## Key Concepts
- 128-bit vector loads (float4 = 4 floats at once)
- Memory coalescing with vector types
- 4× reduction in load instructions
- Alignment requirements for vector loads

## Why float4?
- Global memory has 128-byte sectors
- float4 loads 16 bytes → better utilization
- Fewer instructions → lower latency
- Matches GPU memory system granularity

## Performance Impact
- Reduces memory instruction count by 4×
- Better cache line utilization
- Must ensure proper alignment

## Expected Results
- 5-15% improvement over scalar loads
- Foundation for vectorized GEMM
