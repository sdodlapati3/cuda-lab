# Day 6: Week 21 Performance Baseline

## Learning Objectives
- Consolidate all Week 21 implementations
- Create performance baseline for future comparison
- Understand where we are relative to peak
- Document starting point for optimization journey

## Performance Summary

After Week 21, we have established:

### Implementations Created
1. **Naive GEMM**: One thread per output element
2. **Naive Unrolled**: Loop unrolling optimization
3. **Memory Layout Variants**: Row-major vs column-major analysis
4. **Row-Tiled**: Shared memory for A rows
5. **Column-Tiled**: Shared memory for B columns

### Typical Performance (A100)
| Implementation | TFLOPS | % Peak | % cuBLAS |
|----------------|--------|--------|----------|
| cuBLAS         | ~15    | ~80%   | 100%     |
| Naive          | 0.5-1  | 3-5%   | 5-7%     |
| Naive Unrolled | 0.8-1.5| 5-8%   | 7-10%    |
| Row-Tiled      | 1-2    | 5-10%  | 8-12%    |
| Col-Tiled      | 1-2    | 5-10%  | 8-12%    |

### What We've Learned
1. **Memory is the bottleneck** - not compute
2. **Data reuse is essential** - one-dimensional tiling helps partially
3. **Access patterns matter** - coalescing is critical
4. **cuBLAS is very optimized** - ~80% of theoretical peak

## Week 22 Preview
We'll implement 2D tiling to simultaneously reuse both A and B:
- Shared memory tiles for both matrices
- Dramatic improvement expected (5-10x)
- Path toward 30-40% of peak performance

## Exercises
1. Run comprehensive benchmark
2. Profile with Nsight Compute
3. Identify main bottlenecks
4. Document baseline for comparison
