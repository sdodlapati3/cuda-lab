# Day 6: Optimization Strategy from Roofline

## Learning Objectives

- Choose optimization strategies based on roofline position
- Apply memory-bound vs compute-bound tactics
- Build systematic optimization workflow

## Key Concepts

### The Optimization Decision Tree

```
1. Where is my kernel on the roofline?
   ↓
2. Is it near the ceiling?
   → YES: Achieving hardware limits
   → NO: Room for improvement
   ↓
3. Which ceiling matters?
   → Below memory slope: Memory-bound
   → Below compute ceiling: Compute-bound
   → Low on both: Latency-bound
   ↓
4. Apply appropriate tactics
```

### Memory-Bound Optimizations

| Issue | Solution |
|-------|----------|
| Low bandwidth | Coalesced access |
| Redundant loads | Shared memory caching |
| Random access | Restructure algorithm |
| Low AI | Kernel fusion |

### Compute-Bound Optimizations

| Issue | Solution |
|-------|----------|
| Low occupancy | Reduce registers/shared mem |
| Low ILP | Unroll loops |
| Serial deps | Restructure for parallelism |
| Divergence | Reorganize warps |

### Latency-Bound Optimizations

| Issue | Solution |
|-------|----------|
| Barriers | Reduce sync points |
| Launch overhead | Kernel fusion |
| Memory stalls | Prefetching, more threads |

## Build & Run

```bash
./build.sh
./build/optimization_demo
```
