# Day 2: Register Pressure

## Learning Objectives

- Understand how registers limit occupancy
- Query register usage per kernel
- Control register usage with launch bounds

## Key Concepts

### Register Allocation

```
Total Registers per SM / Registers per Thread = Max Threads per SM

A100: 65536 registers
  32 regs/thread → 2048 threads (100% occupancy)
  64 regs/thread → 1024 threads (50% occupancy)
  128 regs/thread → 512 threads (25% occupancy)
```

### Checking Register Usage

```bash
nvcc --ptxas-options=-v kernel.cu
# Shows: Used XX registers, XX bytes smem
```

### Controlling Registers

```cpp
__global__ __launch_bounds__(256, 4)  // maxThreads, minBlocks
void kernel() { ... }
```

## Build & Run

```bash
./build.sh
./build/register_analysis
```
