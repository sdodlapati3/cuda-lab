# Week 6: Memory Hierarchy

## Overview

Master the CUDA memory hierarchy - the most critical concept for GPU performance.

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Memory Types Overview | Global, shared, registers, constant, texture |
| 2 | Global Memory & Coalescing | Memory transactions, coalesced access patterns |
| 3 | Shared Memory Basics | Block-level cache, bank conflicts |
| 4 | Shared Memory Optimization | Bank conflict resolution, tiling |
| 5 | Registers & Local Memory | Register pressure, spilling |
| 6 | Constant & Texture Memory | Broadcast access, spatial locality |

## Key Mental Models

### Memory Hierarchy

```
Fastest ─┐
         │  Registers (per thread, ~255 max)
         │  Shared Memory (per block, 48-164 KB)
         │  L1 Cache (per SM, automatic)
         │  L2 Cache (chip-wide, automatic)
         │  Global Memory (device RAM, 16-80 GB)
Slowest ─┘
```

### Memory Characteristics

| Type | Scope | Lifetime | Size | Speed |
|------|-------|----------|------|-------|
| Register | Thread | Thread | ~255 per thread | ~1 cycle |
| Shared | Block | Block | 48-164 KB/SM | ~5 cycles |
| L1 Cache | SM | Automatic | 128-192 KB/SM | ~30 cycles |
| L2 Cache | Device | Automatic | 6-80 MB | ~200 cycles |
| Global | Device | App | 16-80 GB | ~500 cycles |
| Constant | Device | App | 64 KB | ~5 cycles (cached) |

### Bandwidth vs Latency

```
Global Memory: High bandwidth (~2 TB/s), High latency (~500 cycles)
Shared Memory: Lower bandwidth, Very low latency (~5 cycles)

Key: Hide global memory latency with many threads
     Use shared memory to reduce global memory traffic
```

## Gate Criteria

- [ ] Explain memory hierarchy with cycle counts
- [ ] Implement coalesced memory access
- [ ] Use shared memory to tile a computation
- [ ] Identify and fix bank conflicts
- [ ] Profile memory bandwidth with Nsight Compute

## Reference: CUDA Programming Guide

- Chapter 5: Memory Hierarchy
- Chapter 5.3: Shared Memory
- Chapter 5.4: Global Memory
- Best Practices Guide: Memory Optimizations

## Common Mistakes

1. **Non-coalesced global access**: Use stride-1 access patterns
2. **Bank conflicts in shared memory**: Pad arrays or change access patterns
3. **Register spilling**: Too many local variables → slow local memory
4. **Not using shared memory**: Many algorithms benefit from tiling
