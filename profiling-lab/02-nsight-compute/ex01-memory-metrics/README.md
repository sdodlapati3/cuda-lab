# Exercise 01: Memory Metrics Analysis

## Learning Objectives
- Understand GPU memory hierarchy metrics
- Measure achieved bandwidth vs theoretical peak
- Identify memory-bound kernels
- Analyze cache hit rates and efficiency

## Background

GPU memory hierarchy affects performance dramatically:

```
Registers (fastest) → L1/Shared Memory → L2 Cache → Global Memory (slowest)
~8 TB/s              ~12 TB/s           ~4 TB/s     ~2 TB/s (A100)
```

### Key Memory Metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| `dram__bytes_read` | Global memory reads | Minimize |
| `dram__bytes_write` | Global memory writes | Minimize |
| `l2_hit_rate` | L2 cache efficiency | Maximize (>80%) |
| `sm__memory_throughput` | Achieved bandwidth | Close to peak |

## Exercise: Profile a Reduction Kernel

### Step 1: Create baseline kernel

```cpp
// naive_reduce.cu
__global__ void naive_reduce(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Naive reduction with divergent warps
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

### Step 2: Profile with Nsight Compute

```bash
# Basic memory metrics
ncu --metrics \
    dram__bytes_read.sum,\
    dram__bytes_write.sum,\
    l2_tex_hit_rate.pct,\
    sm__throughput.avg_pct_of_peak_sustained_elapsed \
    ./reduction

# Full memory analysis
ncu --section MemoryWorkloadAnalysis ./reduction
```

### Step 3: Interpret Results

```
Section: Memory Workload Analysis
----------------------------------
Memory Throughput:     1.5 TB/s (75% of peak)
L2 Hit Rate:          45%
Global Load Efficiency: 50%  ← Problem!
```

**Global Load Efficiency < 100% indicates:**
- Uncoalesced memory access
- Strided access patterns
- Misaligned addresses

### Step 4: Optimize

```cpp
// Optimized: Sequential addressing (coalesced)
__global__ void optimized_reduce(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    // Sequential reduction - better memory access
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

### Step 5: Compare Metrics

```bash
# Profile both versions
ncu --section MemoryWorkloadAnalysis -o naive ./naive_reduction
ncu --section MemoryWorkloadAnalysis -o optimized ./optimized_reduction

# Compare
ncu --compare naive.ncu-rep optimized.ncu-rep
```

## Key Metrics Reference

### Global Memory
```
dram__bytes_read.sum          # Total bytes read from DRAM
dram__bytes_write.sum         # Total bytes written to DRAM
dram__throughput.avg_pct_of_peak  # % of theoretical peak
```

### L2 Cache
```
lts__t_sectors_srcunit_tex_op_read.sum    # L2 sectors read
lts__t_sector_hit_rate.pct                 # L2 hit rate
l2_tex_read_throughput                     # L2 read bandwidth
```

### L1 Cache / Shared Memory
```
sm__inst_executed_op_shared_ld.sum   # Shared memory loads
sm__inst_executed_op_shared_st.sum   # Shared memory stores
l1tex__t_sector_hit_rate.pct         # L1 hit rate
```

## Analysis Questions

1. What is the achieved global memory bandwidth?
2. What is the L2 cache hit rate?
3. Is the kernel memory-bound or compute-bound?
4. How much data is being re-read from DRAM?

## Success Criteria

- [ ] Can profile memory metrics with ncu
- [ ] Understand coalescing impact on bandwidth
- [ ] Improved global load efficiency to >90%
- [ ] Know which metrics indicate memory bottleneck
