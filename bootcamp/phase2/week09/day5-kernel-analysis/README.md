# Day 5: Kernel Analysis with Roofline

## Learning Objectives

- Position kernels on roofline using profiler data
- Identify bound type from metrics
- Compare predicted vs measured performance

## Key Concepts

### Using NCU for Roofline Data

```bash
# Get roofline metrics
ncu --set roofline ./your_app

# Key metrics:
# - sm__sass_thread_inst_executed_op_fadd_pred_on.sum
# - sm__sass_thread_inst_executed_op_fmul_pred_on.sum  
# - dram__bytes.sum
```

### Determining Bound Type

| Metric Pattern | Bound Type |
|---------------|------------|
| DRAM BW near peak, compute low | Memory-bound |
| Compute near peak, DRAM low | Compute-bound |
| Both low | Latency-bound (stalls) |

### Analyzing Phase 1 Kernels

We'll measure kernels from previous weeks:
- Vector add (AI ≈ 0.125, memory-bound)
- Reduction (AI varies with block size)
- Histogram (atomic-limited)
- Transpose (memory-bound, limited by patterns)

## Build & Run

```bash
./build.sh
./build/analyze_kernels

# With NCU profiling:
ncu --set roofline ./build/analyze_kernels
```
