# Day 6: Streams & Events

## Learning Objectives

- Understand asynchronous execution with streams
- Measure kernel timing with events
- Overlap compute and data transfer

## Key Concepts

### Default Stream (Stream 0)

Without explicit streams, everything runs sequentially on the default stream:

```cuda
kernel_A<<<grid, block>>>();  // Wait for A
kernel_B<<<grid, block>>>();  // Then B
cudaMemcpy(...);               // Then copy
```

### Named Streams

Streams enable concurrent execution:

```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel_A<<<grid, block, 0, stream1>>>();  // Run on stream1
kernel_B<<<grid, block, 0, stream2>>>();  // Run on stream2 (concurrent!)

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### CUDA Events

Events for timing and synchronization:

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<...>>>();
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

### Overlapping Compute and Transfer

```cuda
// Async copy + compute overlap
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
kernel<<<..., stream2>>>(d_b);  // Runs while copy proceeds
```

**Requirement**: Host memory must be pinned for async operations!

```cuda
cudaMallocHost(&h_pinned, size);  // Pinned memory
```

## Exercises

1. **Time a kernel**: Use events for accurate GPU timing
2. **Concurrent streams**: Run multiple kernels in parallel
3. **Overlap transfer**: Hide memory copy latency

## Build & Run

```bash
./build.sh
./build/streams_demo
./build/events_timing
```
