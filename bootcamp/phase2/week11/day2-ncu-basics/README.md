# Day 2: NCU Basics

## Learning Objectives

- Use ncu for kernel profiling
- Understand metric collections
- Read the roofline chart

## Key Concepts

### NCU Metric Sets

```bash
# Quick summary
ncu --set default ./app

# Full analysis
ncu --set full ./app

# Roofline data
ncu --set roofline ./app

# Specific metrics
ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed ./app
```

### Key Metrics

| Metric | Meaning |
|--------|---------|
| sm__throughput | SM utilization |
| dram__throughput | Memory bandwidth |
| l2__throughput | L2 cache utilization |
| gpu__compute_memory_throughput | Compute vs memory bound |

### Reading NCU Output

Look for:
- **Roofline position**: Memory or compute bound?
- **SOL %**: % of peak achieved
- **Stall reasons**: Why is the SM waiting?

## Build & Run

```bash
./build.sh
./build/ncu_demo
ncu --set full ./build/ncu_demo
```
