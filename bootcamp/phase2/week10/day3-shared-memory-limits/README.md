# Day 3: Shared Memory Limits

## Learning Objectives

- Understand how shared memory limits occupancy
- Configure dynamic shared memory
- Balance shared memory vs occupancy

## Key Concepts

### Shared Memory Partitioning

```
Shared Memory per SM / Shared Memory per Block = Max Blocks per SM

A100: 164KB configurable per SM
  If block uses 48KB → max 3 blocks/SM
  If block uses 16KB → max 10 blocks/SM
```

### Static vs Dynamic Shared Memory

```cpp
// Static: known at compile time
__shared__ float smem[1024];

// Dynamic: specified at launch
extern __shared__ float smem[];
kernel<<<blocks, threads, dynamic_size>>>(args);
```

### Carveout Configuration

```cpp
// Request more shared memory vs L1 cache
cudaFuncSetAttribute(kernel, 
    cudaFuncAttributeMaxDynamicSharedMemorySize, 
    98304);  // Request 96KB
```

## Build & Run

```bash
./build.sh
./build/shared_mem_limits
```
