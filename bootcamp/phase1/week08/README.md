# Week 8: Synchronization & Atomics

## Overview

Master thread synchronization, atomic operations, and cooperative patterns.

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Barriers & __syncthreads | Block synchronization, fence semantics |
| 2 | Warp-Level Primitives | __shfl, __ballot, __any, __all |
| 3 | Atomic Operations | atomicAdd, atomicCAS, atomicExch |
| 4 | Memory Fences | __threadfence, visibility, ordering |
| 5 | Cooperative Groups | Flexible synchronization, tiled partitions |
| 6 | Lock-Free Patterns | Spinlocks, progress guarantees |

## Key Mental Models

### Synchronization Levels

| Level | Scope | Primitive |
|-------|-------|-----------|
| Warp | 32 threads | __syncwarp(), __shfl_sync() |
| Block | All block threads | __syncthreads() |
| Grid | All grid threads | Cooperative groups / atomic |
| Device | Multiple kernels | Streams, events |

### When to Sync

1. **Before reading** data written by other threads
2. **After writing** data to be read by other threads
3. **At reduction boundaries** within shared memory
4. **Before/after** shared memory tile loads

### Common Mistakes

- Missing sync → race condition
- Sync in divergent code → deadlock
- Over-syncing → performance loss

## Gate Criteria

- [ ] Implement parallel reduction without race conditions
- [ ] Use warp primitives for efficient communication
- [ ] Understand memory fence semantics
- [ ] Apply cooperative groups for flexible sync

## Reference: CUDA Programming Guide

- Chapter 5: Hardware Implementation (warps)
- Chapter 7.5: Memory Fence Functions
- Chapter 9: Cooperative Groups
