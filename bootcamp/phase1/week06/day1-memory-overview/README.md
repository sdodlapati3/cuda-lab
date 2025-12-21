# Day 1: Memory Types Overview

## Learning Objectives

- Understand all memory types available in CUDA
- Know when to use each memory type
- Measure memory access performance

## Memory Types Summary

### Register Memory
```cuda
__global__ void kernel() {
    int a = 5;      // Register
    float b = 3.14f; // Register
}
```
- **Fastest** memory
- Private to each thread
- Limited: ~255 registers per thread
- Spills to local memory if exceeded

### Shared Memory
```cuda
__global__ void kernel() {
    __shared__ float smem[256];  // Shared within block
    smem[threadIdx.x] = ...;
}
```
- Fast scratchpad memory
- Shared by all threads in block
- Programmer-managed cache
- 48-164 KB per SM

### Global Memory
```cuda
__global__ void kernel(float* global_data) {
    global_data[idx] = ...;  // Global memory access
}
```
- Main GPU memory (VRAM)
- Accessible by all threads
- High bandwidth, high latency
- Coalescing is critical!

### Constant Memory
```cuda
__constant__ float coefficients[256];  // Constant cache

__global__ void kernel() {
    float c = coefficients[0];  // Broadcast to all threads
}
```
- Read-only cache
- Optimized for broadcast (all threads read same address)
- 64 KB total

### Local Memory
```cuda
__global__ void kernel() {
    float big_array[1000];  // Too big for registers → local memory
}
```
- Per-thread but stored in global memory
- Slow! Used when registers spill
- Avoid large local arrays

## Decision Tree

```
Need to share data between threads in a block?
  → Use shared memory

Read-only data accessed by all threads?
  → Use constant memory

Per-thread temporary data?
  → Use registers (keep variables small)

Large array that doesn't fit in shared/registers?
  → Use global memory (ensure coalescing)
```

## Exercises

1. **Memory type demo**: Access each memory type and print latency
2. **Register pressure**: See what happens with too many variables
3. **Memory comparison**: Benchmark read speeds

## Build & Run

```bash
./build.sh
./build/memory_types
./build/memory_comparison
```
