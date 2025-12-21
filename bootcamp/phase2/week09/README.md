# Week 9: Roofline Model

## Overview

Master the roofline model - the most important mental framework for GPU performance analysis.

| Day | Topic | Focus |
|-----|-------|-------|
| 1 | Arithmetic Intensity | FLOPS per byte definition |
| 2 | Memory Bandwidth | Measuring peak and achieved |
| 3 | Compute Throughput | FLOPS measurement |
| 4 | Building Rooflines | Plotting tools and methods |
| 5 | Kernel Analysis | Positioning real kernels |
| 6 | Optimization Strategy | Actions based on position |

## Prerequisites

- Phase 1 complete (have working kernels to analyze)
- Nsight Compute installed
- Understanding of memory hierarchy

## Key Concepts

### Arithmetic Intensity (AI)
```
AI = FLOPS / Bytes Transferred

Examples:
- Vector add: AI = 1 FLOP / 12 bytes ≈ 0.08  (memory-bound)
- Matrix mul: AI = 2N / 8 bytes ≈ N/4       (compute-bound for large N)
- Reduction:  AI = 1 FLOP / 4 bytes = 0.25  (memory-bound)
```

### The Roofline
```
Attainable FLOPS/s = min(Peak FLOPS/s, Peak BW × AI)
```

### Ridge Point
Where memory and compute ceilings meet:
```
Ridge Point AI = Peak FLOPS/s / Peak BW

A100 example:
  Peak FP32 = 19.5 TFLOPS
  Peak BW = 2039 GB/s
  Ridge = 19.5e12 / 2039e9 ≈ 9.6 FLOPS/byte
```

## Daily Workflow

```bash
cd day{N}-{topic}
chmod +x build.sh
./build.sh
```

## Success Criteria

By end of week, you should:
- [ ] Calculate AI for any kernel
- [ ] Measure peak bandwidth on your GPU
- [ ] Measure peak FLOPS on your GPU
- [ ] Plot roofline for your hardware
- [ ] Position Phase 1 kernels on roofline
- [ ] Know optimization strategy based on position

## Key Insight

> **Most CUDA kernels are memory-bound.**
> 
> If your kernel's AI < ridge point, optimizing compute is wasted effort.
> Focus on memory access patterns first.
