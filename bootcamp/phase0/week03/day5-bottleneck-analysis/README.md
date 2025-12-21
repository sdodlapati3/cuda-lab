# Day 5: Bottleneck Analysis

## What You'll Learn

- Identify memory vs compute bottlenecks
- Use ncu sections for diagnosis
- Interpret performance limiters
- Choose optimization strategies

## The Bottleneck Hierarchy

```
1. Latency Bound   → Not enough parallelism
2. Memory Bound    → Waiting for data
3. Compute Bound   → ALUs are saturated
```

## Quick Diagnosis

### Memory Bound Symptoms
- High DRAM throughput (>70%)
- Low SM throughput (<50%)
- Kernel time scales with data size

### Compute Bound Symptoms
- High SM throughput (>70%)
- Low DRAM throughput (<50%)
- Kernel time scales with compute

### Latency Bound Symptoms
- Low everything (<50%)
- Low occupancy
- Instruction stalls

## Quick Start

```bash
./build.sh

# Identify bottlenecks
ncu --set full ./build/bottleneck_demo

# Quick check
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./build/bottleneck_demo
```

## ncu Speed Of Light Analysis

```
Section: GPU Speed Of Light Throughput
Metric Name              Value    Peak
------------------------------------
DRAM Throughput          75.2%    <-- High = memory bound
SM Throughput            42.1%    <-- Low when mem bound
```

## Bottleneck-Specific Optimizations

### If Memory Bound
1. **Reduce memory traffic**
   - Cache in shared memory
   - Register blocking
   - Kernel fusion

2. **Improve access patterns**
   - Coalesced access
   - Vectorized loads (float4)
   - Aligned access

3. **Use faster memory**
   - L1/shared instead of global
   - Texture cache for read-only

### If Compute Bound
1. **Reduce instruction count**
   - Faster math (__fast_math)
   - Strength reduction
   - Loop unrolling

2. **Use special functions**
   - rsqrtf instead of 1/sqrtf
   - __expf instead of expf

3. **Increase parallelism**
   - ILP (instruction level)
   - More independent work

### If Latency Bound
1. **Increase occupancy**
   - Reduce registers
   - Reduce shared memory
   - Tune block size

2. **Hide latency**
   - More independent operations
   - Prefetching

## Exercises

1. Profile and classify each kernel
2. Apply appropriate optimization
3. Measure improvement
4. Iterate until hitting ceiling
