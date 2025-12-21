# Day 1: What is Occupancy?

## Learning Objectives

- Define occupancy and its components
- Query device occupancy limits
- Calculate theoretical occupancy

## Key Concepts

### Occupancy Definition

```
Occupancy = Active Warps / Max Warps per SM

100% occupancy on A100:
  = 64 warps = 2048 threads per SM
```

### Why Occupancy Matters

GPUs hide latency by switching between warps. More active warps means:
- More work to switch to during memory stalls
- Better latency hiding
- (Usually) better throughput

### What Limits Occupancy

1. **Registers per thread** - Each thread needs registers
2. **Shared memory per block** - Blocks share this resource
3. **Threads per block** - Block size affects scheduling
4. **Max blocks per SM** - Hardware limit

## Build & Run

```bash
./build.sh
./build/occupancy_basics
```
