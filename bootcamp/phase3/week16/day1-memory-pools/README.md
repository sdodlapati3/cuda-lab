# Day 1: Memory Pools

## Learning Objectives

- Use CUDA memory pools
- Implement stream-ordered allocation
- Reduce allocation overhead

## Key Concepts

### The Problem with cudaMalloc

```cpp
// Expensive: ~1ms per call, synchronizes device
for (int i = 0; i < 1000; i++) {
    cudaMalloc(&ptr, size);  // Ouch!
    kernel<<<...>>>(ptr);
    cudaFree(ptr);           // Ouch!
}
```

### Memory Pool Solution

```cpp
// Get default memory pool
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, 0);

// Stream-ordered allocation (~10μs, async)
cudaMallocAsync(&ptr, size, stream);
kernel<<<..., stream>>>(ptr);
cudaFreeAsync(ptr, stream);  // Returns to pool
```

### Pool Configuration

```cpp
// Allow pool to grow
cudaMemPoolSetAttribute(pool, 
    cudaMemPoolAttrReleaseThreshold, 
    &threshold);

// Trim unused memory
cudaMemPoolTrimTo(pool, 0);
```

### Benefits

1. **Reduced latency**: 100x faster than cudaMalloc
2. **No synchronization**: Truly async allocation
3. **Memory reuse**: Freed blocks cached for reuse
4. **Stream-ordered**: Respects execution order

## Build & Run

```bash
./build.sh
./build/memory_pools
```
