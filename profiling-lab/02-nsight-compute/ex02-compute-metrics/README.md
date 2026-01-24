# Exercise 02: Compute Metrics & Occupancy

## Learning Objectives
- Understand occupancy and its impact on performance
- Analyze warp execution efficiency
- Identify compute-bound vs memory-bound kernels
- Tune block size for optimal occupancy

## Background

### Occupancy

**Occupancy** = Active warps / Maximum warps per SM

High occupancy helps hide latency, but isn't always necessary for peak performance.

### Limiting Factors
1. **Registers per thread**: More registers → fewer threads per SM
2. **Shared memory per block**: More shared memory → fewer blocks per SM
3. **Block size**: Must be multiple of 32 (warp size)

## Key Compute Metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| `sm__warps_active.avg.pct_of_peak_sustained` | Achieved occupancy | Depends on kernel |
| `smsp__cycles_active.avg.pct_of_peak_sustained` | SM utilization | High (>80%) |
| `sm__sass_thread_inst_executed_op_*` | Instructions by type | Understand workload |
| `smsp__inst_executed.avg` | Instructions per SM | Baseline |

## Exercise: Analyze Occupancy Impact

### Step 1: Create test kernel

```cpp
// occupancy_test.cu
template<int BLOCK_SIZE>
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Step 2: Profile different block sizes

```bash
# Test occupancy at different block sizes
for bs in 64 128 256 512 1024; do
    ncu --metrics \
        sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
        sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
        gpu__time_duration.sum \
        ./occupancy_test $bs
done
```

### Step 3: Use Occupancy Calculator

```bash
# Get occupancy limiting factors
ncu --section Occupancy ./occupancy_test 256
```

Output example:
```
Occupancy Analysis:
  Theoretical Occupancy: 100%
  Achieved Occupancy:    75%
  
  Limiting Factor: Registers
    Registers per thread: 32
    Max threads with 32 registers: 2048
    
  Block Size Analysis:
    64 threads:  50% occupancy
    128 threads: 75% occupancy
    256 threads: 100% occupancy
    512 threads: 100% occupancy
```

### Step 4: Analyze Warp Efficiency

```bash
# Warp execution metrics
ncu --metrics \
    smsp__warps_launched.sum,\
    smsp__thread_inst_executed_per_inst_executed.ratio,\
    smsp__inst_executed_op_branch.sum,\
    smsp__inst_executed_op_branch_divergent.sum \
    ./kernel_with_divergence
```

**Key insight**: `thread_inst_executed_per_inst_executed` < 32 indicates divergence

## Compute-Bound vs Memory-Bound

### How to determine:

```bash
ncu --section SpeedOfLight ./my_kernel
```

Output:
```
Speed Of Light:
  Memory: 85% of peak  ← Memory bound
  Compute: 25% of peak
  
  # OR
  
  Memory: 30% of peak
  Compute: 90% of peak  ← Compute bound
```

### Optimization strategy:
- **Memory-bound**: Improve memory access patterns, use caching
- **Compute-bound**: Increase parallelism, use tensor cores

## Register Pressure Analysis

```bash
# Check register usage
ncu --metrics \
    launch__registers_per_thread,\
    launch__shared_mem_per_block_static,\
    launch__shared_mem_per_block_dynamic \
    ./my_kernel
```

### Reducing register pressure:
```cpp
// Force max registers per thread
__launch_bounds__(256, 2)  // 256 threads, 2 blocks per SM minimum
__global__ void my_kernel(...) { }
```

## Practical Exercise

### Profile matrix multiplication:

```bash
# Full compute analysis
ncu --section ComputeWorkloadAnalysis ./matmul

# Compare naive vs optimized
ncu -o naive --section SpeedOfLight ./matmul_naive
ncu -o tiled --section SpeedOfLight ./matmul_tiled
ncu --compare naive.ncu-rep tiled.ncu-rep
```

## Analysis Questions

1. What's the achieved occupancy? Is it limiting performance?
2. Is there warp divergence? Where?
3. Is the kernel compute-bound or memory-bound?
4. What's the optimal block size for this kernel?

## Success Criteria

- [ ] Can measure and interpret occupancy
- [ ] Identify limiting factors for occupancy
- [ ] Detect warp divergence from metrics
- [ ] Determine if kernel is compute or memory bound
- [ ] Choose optimal block size based on metrics
