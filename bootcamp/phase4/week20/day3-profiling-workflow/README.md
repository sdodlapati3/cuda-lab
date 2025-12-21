# Day 3: Profiling Workflow

## Learning Objectives
- Master nsys for system-level profiling
- Use ncu for kernel-level analysis
- Identify and fix performance bottlenecks
- Develop an optimization workflow

## Profiling Tools

### Nsight Systems (nsys)
System-wide view: CPU, GPU, memory, APIs
```bash
# Basic timeline profile
nsys profile ./my_app

# With specific options
nsys profile -o profile_output --stats=true ./my_app

# View in GUI
nsys-ui profile_output.nsys-rep
```

### Nsight Compute (ncu)
Deep kernel analysis
```bash
# Profile all kernels
ncu ./my_app

# Profile specific kernel
ncu --kernel-name myKernel ./my_app

# Full metrics
ncu --set full -o kernel_profile ./my_app
```

## Key Metrics to Monitor

### Memory Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| Memory Throughput | GB/s achieved | >80% of peak |
| L2 Hit Rate | Cache effectiveness | >50% for reuse |
| Shared Memory Efficiency | Bank conflicts | 100% |

### Compute Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| Occupancy | Active warps / max | >50% |
| SM Efficiency | Time SMs active | >80% |
| Warp Execution Efficiency | Divergence | >80% |

## Optimization Workflow

```
1. Profile with nsys → Find bottleneck location
       ↓
2. Profile with ncu → Understand bottleneck cause
       ↓
3. Optimize → Apply fix
       ↓
4. Re-profile → Verify improvement
       ↓
   Repeat until satisfied
```

## Common Bottlenecks

1. **Memory-bound**: Low compute, high memory traffic
   - Fix: Caching, coalescing, reduce transfers

2. **Compute-bound**: High ALU usage, low memory
   - Fix: Algorithm improvement, instruction-level opt

3. **Latency-bound**: Low occupancy, stalls
   - Fix: More threads, instruction-level parallelism

4. **Transfer-bound**: Too much CPU↔GPU traffic
   - Fix: Pinned memory, batching, streams

## Exercises
1. Profile your code from previous weeks
2. Identify the top bottleneck
3. Apply optimization
4. Document before/after metrics
