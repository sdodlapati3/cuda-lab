# Day 3: Memory Metrics

## Learning Objectives

- Analyze memory bandwidth utilization
- Understand cache behavior
- Diagnose memory inefficiencies

## Key Memory Metrics

### DRAM Metrics
```bash
ncu --metrics dram__bytes.sum,dram__throughput.avg_pct_of_peak_sustained_elapsed ./app
```

| Metric | Meaning |
|--------|---------|
| dram__bytes.sum | Total DRAM bytes transferred |
| dram__throughput | % of peak bandwidth |
| lts__t_sectors | L2 cache sectors |
| l1tex__t_sectors | L1 cache sectors |

### Cache Metrics
```bash
ncu --metrics l2_hit_rate,l1_hit_rate ./app
```

### Memory Efficiency
```bash
ncu --metrics sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared,gld_efficiency ./app
```

## Build & Run

```bash
./build.sh
./build/memory_metrics
ncu --set memory ./build/memory_metrics
```
