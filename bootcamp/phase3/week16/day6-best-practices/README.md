# Day 6: Memory Best Practices

## Learning Objectives

- Apply production memory patterns
- Avoid common memory pitfalls
- Build robust memory management

## Key Concepts

### Memory Allocation Checklist

| Pattern | When to Use | Latency |
|---------|-------------|---------|
| `cudaMalloc` | One-time setup | ~1ms |
| `cudaMallocAsync` | Frequent alloc/free | ~10μs |
| `cudaMallocHost` | Async transfers needed | ~1ms |
| Custom pool | Hot path, fixed sizes | ~1μs |

### Best Practices

1. **Allocate early, reuse often**
   ```cpp
   // Bad: allocate in loop
   for (...) { cudaMalloc(&ptr, size); ... cudaFree(ptr); }
   
   // Good: allocate once
   cudaMalloc(&ptr, max_size);
   for (...) { use(ptr); }
   cudaFree(ptr);
   ```

2. **Use memory pools for variable allocations**
   ```cpp
   cudaMallocAsync(&ptr, size, stream);
   // Returns to pool, not OS
   cudaFreeAsync(ptr, stream);
   ```

3. **Align allocations**
   ```cpp
   size_t aligned_size = ((size + 255) / 256) * 256;
   ```

4. **Profile memory usage**
   ```cpp
   size_t free, total;
   cudaMemGetInfo(&free, &total);
   ```

### Common Pitfalls

- ❌ Allocating in kernel launch loop
- ❌ Forgetting to free on error paths
- ❌ Using pageable memory with async
- ❌ Over-allocating pinned memory

## Build & Run

```bash
./build.sh
./build/memory_best_practices
```
