# Day 2: Memory Metrics

## What You'll Learn

- Measure memory bandwidth utilization
- Understand cache hierarchy metrics
- Identify memory access patterns
- Optimize memory throughput

## Key Memory Metrics

### DRAM (Global Memory)
```
dram__throughput.avg.pct_of_peak_sustained_elapsed  # % of peak bandwidth
dram__bytes.sum                                      # Total bytes transferred
dram__bytes_read.sum                                 # Read bytes
dram__bytes_write.sum                                # Write bytes
```

### L2 Cache
```
lts__t_bytes.sum                     # Total L2 traffic
lts__t_sectors_hit.sum               # L2 hit sectors
lts__t_sectors_miss.sum              # L2 miss sectors
lts__t_sector_hit_rate.pct           # L2 hit rate
```

### L1/Texture Cache
```
l1tex__t_bytes.sum                   # Total L1 traffic
l1tex__t_sectors_hit.sum             # L1 hit sectors
l1tex__t_sectors_miss.sum            # L1 miss sectors
```

## Quick Start

```bash
./build.sh

# Profile memory operations
ncu --set memory -o memory_report ./build/memory_patterns

# Check specific metrics
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./build/memory_patterns
```

## Memory Access Patterns

### Coalesced Access (Good)
```
Thread 0 → addr 0
Thread 1 → addr 4
Thread 2 → addr 8
...
→ Single memory transaction
```

### Strided Access (Bad)
```
Thread 0 → addr 0
Thread 1 → addr 128
Thread 2 → addr 256
...
→ Multiple transactions
```

### Random Access (Worst)
```
Thread 0 → addr 1000
Thread 1 → addr 50
Thread 2 → addr 7500
...
→ Many transactions
```

## Metrics to Watch

| Metric | Good | Bad | Fix |
|--------|------|-----|-----|
| DRAM Throughput | >70% | <30% | Coalesce, vectorize |
| L2 Hit Rate | >90% | <50% | Improve locality |
| Sectors/Request | 1 | >4 | Fix access pattern |

## Exercises

1. Profile coalesced vs strided access
2. Measure the impact of alignment
3. Compare cached vs streaming stores
4. Calculate effective bandwidth
