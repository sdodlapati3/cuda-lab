# Day 3: Compute Metrics

## What You'll Learn

- Measure occupancy and its impact
- Understand warp efficiency
- Analyze instruction throughput
- Optimize compute utilization

## Key Compute Metrics

### Occupancy
```
sm__warps_active.avg.pct_of_peak_sustained_active    # Achieved occupancy %
sm__maximum_warps_per_active_cycle                   # Theoretical max warps
sm__warps_active.avg.per_cycle_active                # Active warps/cycle
```

### Warp Efficiency
```
smsp__thread_inst_executed_per_inst_executed.ratio   # Active threads ratio
smsp__warps_issue_stalled.avg                        # Stalled warps
smsp__inst_executed.sum                              # Instructions executed
```

### Instruction Mix
```
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum  # FP add instructions
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum  # FP mul instructions
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum  # FP FMA instructions
```

## Quick Start

```bash
./build.sh

# Compute workload analysis
ncu --set compute -o compute_report ./build/occupancy_test

# Specific occupancy metrics
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./build/occupancy_test
```

## Understanding Occupancy

### What is Occupancy?
```
Occupancy = Active Warps per SM / Maximum Warps per SM
```

### Occupancy Limiters
1. **Registers per thread** - More registers → fewer threads
2. **Shared memory per block** - More shared mem → fewer blocks
3. **Block size** - Must be multiple of 32, max 1024

### Example Calculation (A100)
```
Max warps per SM: 64 (2048 threads)
Your kernel: 256 threads/block, 32 regs/thread

Registers: 256 × 32 = 8192 regs/block
Available: 65536 regs/SM
Max blocks by regs: 65536 / 8192 = 8 blocks
Warps: 8 blocks × 8 warps = 64 warps → 100% occupancy
```

## Warp Divergence

### Divergent Code
```cpp
if (threadIdx.x < 16) {
    // Half the warp takes this path
} else {
    // Other half takes this path
}
// Both paths execute serially!
```

### Metrics
```
smsp__thread_inst_executed_per_inst_executed.ratio
// Ratio < 32 indicates divergence
```

## Exercises

1. Measure occupancy with different block sizes
2. Find the register pressure of your kernels
3. Identify warp divergence
4. Optimize instruction mix
