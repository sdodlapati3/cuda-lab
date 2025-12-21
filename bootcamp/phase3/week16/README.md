# Week 16: Memory Management

Master GPU memory allocation strategies for production applications.

## Daily Breakdown

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Memory Pools | cudaMemPool, stream-ordered allocation |
| 2 | Pinned Memory | Page-locked host memory, async transfers |
| 3 | Zero-Copy & Mapped | Unified addressing, when to use |
| 4 | Memory Compaction | Defragmentation strategies |
| 5 | Large Data Handling | Chunking, streaming, out-of-core |
| 6 | Best Practices | Production memory patterns |

## Mental Model

```
Memory Allocation Hierarchy:

cudaMalloc (synchronous, expensive)
    ↓ cache allocations
cudaMemPool (stream-ordered, reusable)
    ↓ custom control
Custom allocators (sub-allocation)

Transfer Optimization:

Pageable → Device (slow, staged)
Pinned → Device (fast, DMA)
Mapped (zero-copy, no transfer needed)
```

## Key Metrics

| Allocation Type | Latency | When to Use |
|-----------------|---------|-------------|
| cudaMalloc | ~1ms | Infrequent, large allocs |
| cudaMemPool | ~10μs | Frequent, varying sizes |
| Custom pool | ~1μs | Hot path, fixed sizes |

## Quick Reference

```cpp
// Enable memory pool
cudaMemPool_t pool;
cudaDeviceGetDefaultMemPool(&pool, device);

// Stream-ordered allocation
cudaMallocAsync(&ptr, size, stream);
cudaFreeAsync(ptr, stream);

// Pinned memory
cudaMallocHost(&h_ptr, size);
cudaMemcpyAsync(d_ptr, h_ptr, size, cudaMemcpyHostToDevice, stream);
```
