# Nsight Compute Key Metrics Cheatsheet

Quick reference for the most important metrics and what they mean.

## Speed of Light Metrics

These tell you how close you are to hardware limits.

```bash
ncu --section SpeedOfLight ./kernel
```

| Metric | Meaning | Action if Low |
|--------|---------|---------------|
| `sm__throughput.avg.pct_of_peak_sustained` | Compute utilization | Increase parallelism |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained` | Memory bandwidth utilization | Improve access patterns |
| `l1tex__throughput.avg.pct_of_peak_sustained` | L1 cache throughput | Better locality |

### Interpretation:
- **Both low**: Latency bound (increase occupancy)
- **Memory high, compute low**: Memory bound (optimize access)
- **Compute high, memory low**: Compute bound (optimize arithmetic)

---

## Memory Metrics

### Global Memory (DRAM)
```bash
ncu --metrics dram__bytes.sum,dram__throughput.avg_pct_of_peak ./kernel
```

| Metric | Meaning |
|--------|---------|
| `dram__bytes_read.sum` | Total DRAM reads |
| `dram__bytes_write.sum` | Total DRAM writes |
| `dram__throughput.avg.pct_of_peak` | % of peak bandwidth |

### Memory Efficiency
```bash
ncu --metrics \
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld,\
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st \
    ./kernel
```

| Metric | Ideal | Meaning |
|--------|-------|---------|
| `global_load_efficiency` | 100% | Bytes requested / bytes transferred |
| `global_store_efficiency` | 100% | Same for stores |

**< 100% means uncoalesced access!**

---

## Cache Metrics

### L2 Cache
```bash
ncu --metrics lts__t_sector_hit_rate.pct ./kernel
```

| Metric | Good Value | Meaning |
|--------|------------|---------|
| `lts__t_sector_hit_rate.pct` | >80% | L2 cache hit rate |
| `l2_read_throughput` | High | L2 → SM bandwidth |

### L1 / Shared Memory
```bash
ncu --metrics l1tex__t_sector_hit_rate.pct ./kernel
```

| Metric | Meaning |
|--------|---------|
| `l1tex__t_sector_hit_rate.pct` | L1 cache hit rate |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | Shared memory bank conflicts |

---

## Occupancy Metrics

```bash
ncu --section Occupancy ./kernel
```

| Metric | Meaning |
|--------|---------|
| `sm__warps_active.avg.pct_of_peak_sustained` | Achieved occupancy |
| `sm__maximum_warps_per_active_cycle` | Theoretical max warps |
| `launch__registers_per_thread` | Registers used per thread |
| `launch__shared_mem_per_block` | Shared memory per block |

### Occupancy Limiters:
- **Registers**: Reduce register count with `__launch_bounds__`
- **Shared memory**: Reduce or use dynamic allocation
- **Block size**: Must be multiple of 32

---

## Compute Metrics

### Instruction Mix
```bash
ncu --metrics \
    sm__inst_executed_pipe_alu.sum,\
    sm__inst_executed_pipe_fma.sum,\
    sm__inst_executed_pipe_tensor.sum \
    ./kernel
```

| Metric | Meaning |
|--------|---------|
| `sm__inst_executed_pipe_alu` | Integer/logic operations |
| `sm__inst_executed_pipe_fma` | Floating-point FMA |
| `sm__inst_executed_pipe_tensor` | Tensor Core operations |

### Warp Efficiency
```bash
ncu --metrics \
    smsp__thread_inst_executed_per_inst_executed.ratio \
    ./kernel
```

| Metric | Ideal | Meaning |
|--------|-------|---------|
| Thread efficiency ratio | 32 | Active threads per instruction |

**< 32 indicates warp divergence**

---

## Quick Diagnosis Commands

```bash
# "What's wrong with my kernel?"
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./kernel

# "Is it memory or compute bound?"
ncu --section SpeedOfLight ./kernel

# "Why is memory slow?"
ncu --section MemoryWorkloadAnalysis ./kernel

# "Why is occupancy low?"
ncu --section Occupancy ./kernel

# "Are there bank conflicts?"
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared ./kernel

# "Show me everything"
ncu --set full -o full_report ./kernel
```

---

## Roofline Quick Reference

```bash
# Generate roofline data
ncu --section SpeedOfLight_RooflineChart -o roofline ./kernel
```

### Reading the Roofline:
- **X-axis**: Arithmetic Intensity (FLOPs/Byte)
- **Y-axis**: Performance (GFLOPS)
- **Roofline**: Theoretical maximum at each AI
- **Your kernel**: Dot below the line

### Position Meaning:
| Position | Interpretation |
|----------|----------------|
| Left of knee | Memory bound |
| Right of knee | Compute bound |
| Near line | Well optimized |
| Far below line | Optimization opportunity |

---

## Metric Collection Shortcuts

```bash
# Save metrics to file
ncu --csv --metrics <metrics> ./kernel > results.csv

# Compare two kernels
ncu -o baseline ./baseline
ncu -o optimized ./optimized
ncu --compare baseline.ncu-rep optimized.ncu-rep

# Profile specific kernel (by name)
ncu --kernel-name "my_kernel" ./app

# Profile specific kernel (by index)
ncu --kernel-id 0 ./app

# Limit iterations
ncu --launch-count 10 ./app
```
