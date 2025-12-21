# Week 11: Profiling Mastery

## Theme: Systematic Performance Analysis with NVIDIA Tools

This week builds proficiency with NCU and Nsight Systems to diagnose
performance issues and guide optimization decisions.

## Daily Breakdown

| Day | Topic | Focus |
|-----|-------|-------|
| 1 | NSight Systems Overview | Timeline, kernel/memory overlap |
| 2 | NCU Basics | Metrics collection, roofline |
| 3 | Memory Metrics | Bandwidth, cache analysis |
| 4 | Compute Metrics | FLOPS, instruction mix |
| 5 | Finding Bottlenecks | Systematic diagnosis |
| 6 | Optimization Cycle | Profile-optimize-verify loop |

## Key Tools

### Nsight Systems (nsys)
- **Purpose**: System-wide timeline analysis
- **Use for**: Kernel overlap, memory transfers, CPU-GPU interaction
- **Command**: `nsys profile ./app`

### Nsight Compute (ncu)
- **Purpose**: Detailed kernel analysis
- **Use for**: Per-kernel metrics, roofline, bottleneck identification
- **Command**: `ncu --set full ./app`

## Mental Model: The Profiling Hierarchy

```
1. Nsight Systems (forest view)
   - Are kernels overlapping?
   - Is data transfer a bottleneck?
   - What's the CPU doing?

2. Nsight Compute (tree view)
   - Is this kernel efficient?
   - What limits performance?
   - How far from peak?
```

## Prerequisites
- Week 9-10: Roofline and Occupancy concepts
- Phase 1: Kernel writing experience

## Building
Each day has working examples but many exercises are profile-focused:
```bash
cd dayX-topic
./build.sh
# Then use profilers on the executables
```
