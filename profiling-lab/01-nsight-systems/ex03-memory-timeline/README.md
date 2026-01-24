# Exercise 03: Memory Transfer Timeline Analysis

## Learning Objectives
- Profile and optimize Host-to-Device (H2D) and Device-to-Host (D2H) transfers
- Understand pinned vs pageable memory performance
- Identify memory transfer bottlenecks
- Calculate achieved vs theoretical bandwidth

## Background

Memory transfers between CPU and GPU are often the bottleneck in GPU applications. Understanding transfer patterns is critical for optimization.

### Memory Types
| Type | Allocation | Transfer Speed | Async Capable |
|------|------------|----------------|---------------|
| Pageable | `malloc()` | Slow | No |
| Pinned | `cudaMallocHost()` | Fast | Yes |
| Managed | `cudaMallocManaged()` | Auto-migrated | Depends |

### Theoretical Bandwidth
- PCIe 3.0 x16: ~16 GB/s
- PCIe 4.0 x16: ~32 GB/s
- PCIe 5.0 x16: ~64 GB/s
- NVLink (A100): ~600 GB/s bidirectional

## Exercise Files

```
ex03-memory-timeline/
├── pageable.cu        # Using malloc (slow)
├── pinned.cu          # Using cudaMallocHost (fast)
├── analysis.py        # Parse nsys output
├── Makefile
└── README.md
```

## Part 1: Profile Pageable Memory

```bash
make pageable
nsys profile --stats=true -o pageable ./pageable
```

### What to observe:
1. Transfer times in the timeline
2. Look for "Pageable" memory operations
3. Note the bandwidth achieved

## Part 2: Profile Pinned Memory

```bash
make pinned
nsys profile --stats=true -o pinned ./pinned
```

### Compare:
1. Transfer times should be ~2-3x faster
2. Async operations now possible
3. Look for overlap with compute

## Part 3: Calculate Bandwidth Efficiency

```python
# From nsys stats output:
# Data transferred (bytes) / Time (seconds) = Bandwidth (GB/s)

data_size_gb = 1.0  # 1 GB transfer
transfer_time_ms = 50  # from profiler

achieved_bw = data_size_gb / (transfer_time_ms / 1000)  # GB/s
theoretical_bw = 16  # PCIe 3.0 x16
efficiency = achieved_bw / theoretical_bw * 100

print(f"Achieved: {achieved_bw:.1f} GB/s ({efficiency:.0f}% of peak)")
```

## Part 4: Advanced - Bidirectional Transfer

Test simultaneous H2D and D2H:

```cpp
// With two copy engines, transfers can overlap
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(h_b, d_b, size, cudaMemcpyDeviceToHost, stream2);
```

## Key Profiler Metrics

| Metric | Location | Meaning |
|--------|----------|---------|
| MemCpy HtoD | CUDA API | Host to Device transfer |
| MemCpy DtoH | CUDA API | Device to Host transfer |
| Throughput | GPU row | Bytes/second achieved |
| Duration | Timeline | Transfer time |

## Analysis Questions

1. What percentage of total runtime is spent on memory transfers?
2. What's the achieved bandwidth vs theoretical maximum?
3. Does using pinned memory enable async overlap?
4. Are H2D and D2H happening concurrently?

## Success Criteria

- [ ] Can identify memory transfer time in profiler
- [ ] Understand pinned vs pageable performance difference
- [ ] Calculate bandwidth efficiency
- [ ] Achieved >80% of theoretical PCIe bandwidth with pinned memory

## Common Issues

1. **Low bandwidth**: Not using pinned memory
2. **No overlap**: Using default stream or sync calls
3. **Transfer dominates**: Consider compute/transfer ratio
4. **Unified memory thrashing**: Check page fault events
