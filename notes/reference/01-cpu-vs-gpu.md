# CPU vs GPU: Why Parallel Computing?

## Introduction

Before diving into CUDA, it's crucial to understand *why* GPUs exist and when to use them.

## The Fundamental Difference

### CPU (Central Processing Unit)
- **Optimized for:** Latency (fast single-thread performance)
- **Design philosophy:** Do one thing as fast as possible
- **Cores:** 4-64 powerful cores
- **Cache:** Large caches (MB per core)
- **Control:** Complex branch prediction, out-of-order execution

### GPU (Graphics Processing Unit)
- **Optimized for:** Throughput (many operations per second)
- **Design philosophy:** Do many things in parallel
- **Cores:** Thousands of simple cores
- **Cache:** Smaller, shared caches
- **Control:** Simple, in-order execution

## Analogy: Shipping Packages

Imagine you need to deliver 10,000 packages:

**CPU Approach (Sports Car):**
- 4 very fast sports cars
- Each can carry 1 package
- Each trip takes 1 minute
- Total time: 10,000 ÷ 4 = 2,500 minutes

**GPU Approach (Fleet of Trucks):**
- 1,000 slower trucks
- Each can carry 10 packages
- Each trip takes 10 minutes
- Total time: 10,000 ÷ (1,000 × 10) × 10 = 10 minutes

The GPU wins when you have **lots of independent work**.

## When to Use GPU

✅ **Good for GPU:**
- Same operation on millions of data points
- Matrix/vector operations
- Image/video processing
- Neural network training/inference
- Scientific simulations
- Monte Carlo methods

❌ **Bad for GPU:**
- Sequential algorithms
- Heavy branching (if/else)
- Small data sets
- Operations with dependencies
- Tasks requiring low latency for single items

## The Bottleneck: Memory Transfer

```
CPU Memory ←→ GPU Memory
   ↑              ↑
   |              |
  CPU            GPU
```

Moving data between CPU and GPU is **slow** (PCIe bus).
The GPU must do enough work to justify the transfer cost.

**Rule of thumb:** If computation time < transfer time, stay on CPU.

## Hardware Architecture

### CPU Architecture
```
┌─────────────────────────────────────────┐
│  CPU Core (optimized for speed)         │
│  ┌─────────────────────────────────┐    │
│  │ ALU  ALU  ALU  ALU              │    │
│  │ (Arithmetic Logic Units)        │    │
│  ├─────────────────────────────────┤    │
│  │ Large L1 Cache (32-64 KB)       │    │
│  │ Large L2 Cache (256 KB - 1 MB)  │    │
│  ├─────────────────────────────────┤    │
│  │ Complex Control Logic           │    │
│  │ - Branch prediction             │    │
│  │ - Out-of-order execution        │    │
│  │ - Speculative execution         │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
                    ×4-64 cores
```

### GPU Architecture (NVIDIA)
```
┌─────────────────────────────────────────────────────────────┐
│  Streaming Multiprocessor (SM)                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ CUDA Core CUDA Core CUDA Core CUDA Core ... (64-128)  │  │
│  │ CUDA Core CUDA Core CUDA Core CUDA Core ...           │  │
│  │ (Simple ALUs - add, multiply, FMA)                    │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │ Shared Memory / L1 Cache (64-164 KB)                  │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │ Simple Control (warp scheduler)                       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         ×40-144 SMs
```

## Key GPU Concepts

### 1. SIMT (Single Instruction, Multiple Threads)
All threads in a **warp** (32 threads) execute the same instruction.

```cpp
if (condition) {
    // Half the warp does this
} else {
    // Other half does this
}
// Both paths execute SERIALLY (divergence)
```

### 2. Massive Parallelism
- Thousands of threads running concurrently
- Threads are cheap to create and switch

### 3. Memory Hierarchy
```
Registers (fastest, per-thread)
    ↓
Shared Memory (fast, per-block)
    ↓
L1/L2 Cache (medium)
    ↓
Global Memory (slow, per-device)
    ↓
CPU Memory (slowest, requires PCIe transfer)
```

### 4. Occupancy
How busy the GPU is. Limited by:
- Threads per SM
- Registers per thread
- Shared memory per block

## Performance Comparison

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Vector add (1M) | 2 ms | 0.1 ms | 20× |
| Matrix mul (1K×1K) | 500 ms | 1 ms | 500× |
| Neural net forward | 100 ms | 2 ms | 50× |
| Sorting (1M) | 100 ms | 5 ms | 20× |

*Note: Speedups vary greatly based on implementation quality*

## Summary

| Aspect | CPU | GPU |
|--------|-----|-----|
| Cores | Few powerful | Many simple |
| Threads | ~64 | ~100,000 |
| Latency | Low | High |
| Throughput | Low | High |
| Cache | Large | Small |
| Best for | Serial, branchy | Parallel, uniform |

## Next Steps

1. Complete Exercise 01: Query your GPU properties
2. Read: [CUDA Platform](../../cuda-programming-guide/01-introduction/cuda-platform.md)
3. Next tutorial: [GPU Architecture](02-gpu-architecture.md)

---

[🏠 Tutorials Home](../README.md) | [Next: GPU Architecture →](02-gpu-architecture.md)
