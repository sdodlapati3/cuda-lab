# Exercise 02: Kernel Overlap & Stream Concurrency

## Learning Objectives
- Understand CUDA streams and concurrent execution
- Use Nsight Systems to visualize kernel overlap
- Identify opportunities for parallelism
- Optimize CPU-GPU overlap

## Background

CUDA operations within the same stream execute sequentially. Operations in different streams can execute concurrently if resources allow. Nsight Systems timeline view makes this visible.

### Key Concepts
- **Stream**: Sequence of operations that execute in order
- **Default stream**: Synchronous with all other streams (stream 0)
- **Concurrent kernels**: Multiple kernels running on GPU simultaneously
- **Async memcpy**: Data transfer overlapping with compute

## Exercise Files

```
ex02-kernel-overlap/
├── sequential.cu      # Baseline: all in default stream
├── overlapped.cu      # Your task: use multiple streams
├── solution.cu        # Reference solution
├── Makefile
└── README.md
```

## Part 1: Profile Sequential Execution

```bash
# Build and profile baseline
make sequential
nsys profile --stats=true -o sequential ./sequential

# Open in Nsight Systems GUI
nsys-ui sequential.nsys-rep
```

### What to observe:
1. All kernels execute one after another
2. Memory transfers block kernel execution
3. GPU utilization has gaps

## Part 2: Implement Stream Overlap

Edit `overlapped.cu` to:

1. Create multiple CUDA streams
2. Distribute work across streams
3. Use async memory operations
4. Ensure proper synchronization

### Hints:
```cpp
// Create streams
cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
}

// Async operations in different streams
cudaMemcpyAsync(d_data[i], h_data[i], size, cudaMemcpyHostToDevice, streams[i]);
kernel<<<grid, block, 0, streams[i]>>>(d_data[i], n);
cudaMemcpyAsync(h_result[i], d_data[i], size, cudaMemcpyDeviceToHost, streams[i]);

// Sync all streams
for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamSynchronize(streams[i]);
}
```

## Part 3: Profile and Compare

```bash
# Profile your implementation
make overlapped
nsys profile --stats=true -o overlapped ./overlapped

# Compare timelines
nsys-ui sequential.nsys-rep overlapped.nsys-rep
```

### Analysis Questions:
1. How much kernel overlap do you see?
2. What's the speedup from concurrent execution?
3. Are memory transfers overlapping with compute?
4. What limits further overlap?

## Part 4: Advanced - Measure Overlap Efficiency

Calculate theoretical vs actual overlap:

```
Theoretical speedup = Sum(individual kernel times) / Time(longest path)
Actual speedup = Sequential time / Overlapped time
Overlap efficiency = Actual speedup / Theoretical speedup
```

## Expected Results

| Metric | Sequential | Overlapped | Improvement |
|--------|------------|------------|-------------|
| Total time | ~100ms | ~40ms | 2.5x |
| GPU idle time | 60% | 15% | 4x reduction |
| Kernel overlap | 0% | 75% | - |

## Common Pitfalls

1. **Using default stream**: Operations sync with everything
2. **Not using pinned memory**: Async memcpy falls back to sync
3. **Insufficient work**: Small kernels don't benefit from overlap
4. **Resource contention**: Too many streams can hurt performance

## Profiler Commands Reference

```bash
# Detailed kernel analysis
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true -o analysis ./overlapped

# Export to JSON for scripting
nsys stats --report cuda_gpu_kern_sum --format json analysis.nsys-rep

# Compare two runs
nsys compare sequential.nsys-rep overlapped.nsys-rep
```

## Success Criteria

- [ ] Kernels from different streams overlap in timeline
- [ ] Memory transfers overlap with kernel execution
- [ ] Achieved 2x+ speedup over sequential
- [ ] Can explain why overlap is limited (if any)

## Next Steps

After completing this exercise:
- [ex03-memory-timeline](../ex03-memory-timeline/) - Analyze H2D/D2H transfers
- [ex04-multi-gpu-timeline](../ex04-multi-gpu-timeline/) - NCCL communication profiling
