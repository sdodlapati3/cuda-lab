# Week 23, Day 6: Register Blocking Summary

## Objective
Compare all register blocking strategies and establish best practices.

## Week 23 Summary
- Day 1: Register blocking fundamentals, 4×4 outer product
- Day 2: 2×2 thread tile (1.0 FMA/load)
- Day 3: 4×4 thread tile (2.0 FMA/load)
- Day 4: 8×8 thread tile (4.0 FMA/load)
- Day 5: Register pressure analysis

## Performance Progression
| Config | Outputs/Thread | Registers | Intensity | Expected % |
|--------|---------------|-----------|-----------|------------|
| 1×1    | 1             | ~4        | 0.5       | 10-15%     |
| 2×2    | 4             | ~8        | 1.0       | 20-25%     |
| 4×4    | 16            | ~24       | 2.0       | 30-40%     |
| 8×8    | 64            | ~80       | 4.0       | 40-50%     |

## Key Insights
1. Larger tiles increase compute intensity
2. Register pressure limits practical tile size
3. Occupancy trade-off matters
4. 4×4 or 8×8 typically optimal for FP32 GEMM
