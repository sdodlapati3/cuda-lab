# Phase 0 - Week 3: Performance Analysis

## Overview

Master GPU performance analysis with Nsight Compute and metrics-driven optimization.

| Day | Topic | Tools |
|-----|-------|-------|
| 1 | Nsight Compute Basics | ncu, ncu-ui |
| 2 | Memory Metrics | Bandwidth, cache, throughput |
| 3 | Compute Metrics | Occupancy, warp efficiency |
| 4 | Roofline Analysis | Arithmetic intensity |
| 5 | Bottleneck Analysis | Memory vs compute bound |
| 6 | Optimization Workflow | Iterative improvement |

## Prerequisites

- Week 2 complete (debugging tools)
- Nsight Compute installed (part of CUDA Toolkit)
- Understanding of GPU architecture basics

## Daily Workflow

```bash
cd day{N}-{topic}
chmod +x build.sh
./build.sh
```

## Tool Quick Reference

### Nsight Compute (ncu)
```bash
# Basic profiling
ncu ./my_app

# Save to file
ncu -o report ./my_app

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./my_app

# Full analysis
ncu --set full -o full_report ./my_app

# Specific kernel
ncu --kernel-name my_kernel ./my_app

# Launch range
ncu --launch-skip 10 --launch-count 5 ./my_app
```

### Key Metric Categories
```
Memory:
  - dram__throughput.avg.pct_of_peak_sustained_elapsed
  - l1tex__t_bytes.sum
  - lts__t_bytes.sum

Compute:
  - sm__throughput.avg.pct_of_peak_sustained_elapsed
  - sm__warps_active.avg.pct_of_peak_sustained_active

Occupancy:
  - sm__warps_active.avg.per_cycle_active
  - sm__maximum_warps_per_active_cycle
```

## Week 3 Goals

By the end of this week, you should:
1. ✅ Profile kernels with Nsight Compute
2. ✅ Interpret memory bandwidth metrics
3. ✅ Analyze occupancy and warp efficiency
4. ✅ Use roofline model for analysis
5. ✅ Identify memory vs compute bottlenecks
6. ✅ Apply iterative optimization workflow

## Directory Structure

```
week03/
├── README.md
├── day1-ncu-basics/
├── day2-memory-metrics/
├── day3-compute-metrics/
├── day4-roofline/
├── day5-bottleneck-analysis/
└── day6-optimization-workflow/
```
