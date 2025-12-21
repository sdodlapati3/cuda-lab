# Week 24: Vectorized Memory Access

## Overview
This week optimizes memory access patterns using vector loads
(float4, LDG.128) to maximize memory bandwidth utilization.

## Daily Schedule

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 1 | Vector Loads Intro | float4, LDG.128 instructions |
| 2 | Coalesced Vector Access | Aligning loads for vectorization |
| 3 | Global to Shared | Vectorized tile loading |
| 4 | Shared to Register | Optimizing register loads |
| 5 | Memory Bound Analysis | Bandwidth utilization profiling |
| 6 | Integrated Optimization | Combined techniques |

## Key Concept: Vector Loads

Single 128-bit load vs four 32-bit loads:
```cpp
// 4 separate loads
float a = A[i];
float b = A[i+1];
float c = A[i+2];
float d = A[i+3];

// One vector load
float4 vec = reinterpret_cast<float4*>(&A[i])[0];
```

Benefits:
- Fewer memory transactions
- Better coalescing
- Reduced instruction count

## Expected Performance
After this week: 55-70% of cuBLAS performance
