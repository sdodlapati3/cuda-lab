# Day 6: Optimization Workflow

## What You'll Learn

- Systematic performance optimization process
- Iterative measurement and improvement
- Documentation and tracking
- Complete profiling toolkit

## The Optimization Workflow

```
┌─────────────────────────────────────────────────┐
│  1. BASELINE                                    │
│     - Measure current performance               │
│     - Record all metrics                        │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  2. PROFILE                                     │
│     - Identify bottleneck (mem/compute/latency) │
│     - Find hotspots                             │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  3. HYPOTHESIZE                                 │
│     - Choose optimization based on bottleneck   │
│     - Estimate expected improvement             │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  4. IMPLEMENT                                   │
│     - Make single change                        │
│     - Keep old version for comparison           │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  5. MEASURE                                     │
│     - Compare to baseline                       │
│     - Check if hypothesis was correct           │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  6. ITERATE                                     │
│     - If improved: new baseline                 │
│     - If not: try different approach            │
│     - Continue until satisfied                  │
└─────────────────────────────────────────────────┘
```

## Quick Start

```bash
./build.sh

# Run complete optimization demo
./build/optimization_workflow
```

## Optimization Checklist

### Before Starting
- [ ] Have a correctness test
- [ ] Establish baseline metrics
- [ ] Document initial performance

### Memory Optimizations
- [ ] Coalesced access patterns
- [ ] Vectorized loads (float4)
- [ ] Shared memory caching
- [ ] Aligned allocations
- [ ] Minimize host-device transfers

### Compute Optimizations
- [ ] Fast math intrinsics
- [ ] Loop unrolling
- [ ] Instruction-level parallelism
- [ ] Reduce divergence
- [ ] Use FMA when possible

### Occupancy Optimizations
- [ ] Tune block size
- [ ] Reduce register usage
- [ ] Balance shared memory
- [ ] Minimize synchronization

### System-Level Optimizations
- [ ] Overlap compute and transfer
- [ ] Use streams for concurrency
- [ ] Pinned memory for transfers
- [ ] Consider CUDA graphs

## Profiling Commands Reference

```bash
# System-level timeline
nsys profile -o timeline ./app
nsys stats timeline.nsys-rep

# Kernel-level detailed
ncu --set full -o kernel_report ./app

# Specific metrics
ncu --metrics METRIC_LIST ./app

# Quick bottleneck check
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./app

# Memory analysis
ncu --set memory ./app

# Compute analysis
ncu --set compute ./app

# Roofline
ncu --set roofline ./app
```

## Performance Log Template

```
Date: YYYY-MM-DD
Kernel: kernel_name
Version: vN

Baseline:
  Time: X.XX ms
  DRAM Throughput: XX%
  SM Throughput: XX%
  Bottleneck: [memory/compute/latency]

Optimization: [description]
Expected improvement: XX%

After:
  Time: X.XX ms (Xspeedup)
  DRAM Throughput: XX%
  SM Throughput: XX%

Notes: [observations]
```

## Exercises

1. Follow the complete workflow on the demo kernel
2. Document each optimization step
3. Compare multiple approaches
4. Build your optimization intuition
