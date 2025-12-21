# Day 1: Nsight Compute Basics

## What You'll Learn

- Profile kernels with Nsight Compute (ncu)
- Navigate the profiler output
- Understand key metrics sections
- Compare kernel performance

## Why Nsight Compute?

- **Kernel-level** analysis (vs Nsight Systems for system-level)
- **Detailed metrics** for each kernel launch
- **Roofline** visualization
- **Source correlation** to see hot spots

## Quick Start

```bash
./build.sh

# Basic profiling
ncu ./build/vector_ops

# Save report
ncu -o vector_report ./build/vector_ops

# View in GUI
ncu-ui vector_report.ncu-rep
```

## Key ncu Commands

### Basic Usage
```bash
# Profile all kernels
ncu ./my_app

# Profile specific kernel
ncu --kernel-name add_kernel ./my_app

# Skip first N launches (warmup)
ncu --launch-skip 5 ./my_app

# Profile only N launches
ncu --launch-count 3 ./my_app
```

### Metric Sets
```bash
# Quick overview (default)
ncu ./my_app

# Full analysis (slower, more metrics)
ncu --set full ./my_app

# Memory-focused
ncu --set memory ./my_app

# Compute-focused
ncu --set compute ./my_app
```

### Specific Metrics
```bash
# Single metric
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./my_app

# Multiple metrics
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed ./my_app
```

## Understanding the Output

### Speed Of Light (SOL) Section
```
Section: GPU Speed Of Light Throughput
----------------------- ------------- ----------
Metric Name               Metric Unit Metric Value
----------------------- ------------- ----------
DRAM Throughput                    %        75.2
SM Throughput                      %        42.1
```
- **DRAM Throughput**: How much of peak memory bandwidth used
- **SM Throughput**: How much of peak compute used

### Memory Workload Analysis
```
Section: Memory Workload Analysis
----------------------- ------------- ----------
L1/TEX Cache Throughput            %        85.3
L2 Cache Throughput                %        62.1
DRAM Throughput                    %        75.2
```

### Compute Workload Analysis
```
Section: Compute Workload Analysis
----------------------- ------------- ----------
Achieved Occupancy                 %        75.0
Theoretical Occupancy              %       100.0
```

## Exercises

1. Profile the vector operations and identify the fastest
2. Compare bandwidth utilization across kernels
3. Find which kernel is memory-bound vs compute-bound
4. Practice using different metric sets
