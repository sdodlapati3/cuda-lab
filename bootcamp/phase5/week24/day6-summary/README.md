# Week 24, Day 6: Vectorization Summary

## Objective
Review all vectorization techniques and their combined impact on GEMM.

## Week 24 Summary

| Day | Technique | Benefit |
|-----|-----------|---------|
| 1 | float4 loads | 4× fewer load instructions |
| 2 | Transposed B | Coalesced column access |
| 3 | Async copy | Overlap load and compute |
| 4 | Swizzled smem | Zero bank conflicts |
| 5 | Combined | All techniques together |

## Performance Impact
- Individual techniques: 5-15% each
- Combined: 40-60% of cuBLAS (from ~30%)

## Key Insights
1. Memory access patterns dominate performance
2. Vector loads reduce instruction count
3. Async copy hides memory latency
4. Bank conflict avoidance is essential

## Next Steps
Week 25: Warp-level optimizations
