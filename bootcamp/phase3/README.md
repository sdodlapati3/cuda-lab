# Phase 3: Production Patterns (Weeks 13-16)

## Overview

Phase 3 teaches you production-ready CUDA patterns used in real-world high-performance code.

## Prerequisites

Before starting Phase 3, you should be able to:
- Calculate arithmetic intensity and use roofline analysis
- Optimize occupancy through resource management
- Profile with nsys and ncu to find bottlenecks
- Apply latency hiding techniques (ILP, TLP, prefetching)

## Weekly Themes

### Week 13: Warp-Level Programming
Master warp-level primitives for efficient intra-warp communication.

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Warp Fundamentals | Warp size, SIMT model, warp divergence |
| 2 | Shuffle Instructions | `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync` |
| 3 | Warp Reductions | Efficient reduction without shared memory |
| 4 | Warp Scans | Inclusive/exclusive prefix sums within warp |
| 5 | Warp Vote Functions | `__ballot_sync`, `__any_sync`, `__all_sync` |
| 6 | Warp-Level Patterns | Building blocks for complex algorithms |

### Week 14: CUDA Libraries
Leverage optimized libraries instead of reinventing the wheel.

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | cuBLAS Basics | Matrix operations, GEMM |
| 2 | cuBLAS Advanced | Batched operations, tensor cores |
| 3 | CUB Primitives | Device-wide and block-level algorithms |
| 4 | CUB Advanced | Custom operators, segmented operations |
| 5 | Thrust | High-level parallel algorithms |
| 6 | Library Selection | When to use which library |

### Week 15: Kernel Fusion
Combine operations to reduce memory traffic.

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Fusion Basics | Why fuse, what to fuse |
| 2 | Element-wise Fusion | Combining pointwise operations |
| 3 | Reduction Fusion | Fused map-reduce patterns |
| 4 | Tiled Fusion | Fusing with shared memory |
| 5 | Producer-Consumer | Connecting pipeline stages |
| 6 | Fusion Strategies | Real-world fusion decisions |

### Week 16: Memory Management Patterns
Production-grade memory handling.

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Memory Pools | cudaMallocAsync, custom pools |
| 2 | Pinned Memory | Host memory for fast transfers |
| 3 | Zero-Copy Memory | Mapped host memory |
| 4 | Memory Compaction | Defragmentation strategies |
| 5 | Large Data Patterns | Streaming, chunking |
| 6 | Memory Best Practices | Production patterns |

## Mental Models for Phase 3

### Warp Thinking
```
32 threads move in lockstep
Shuffle = FREE communication within warp
Shared memory = communication BETWEEN warps
```

### Library vs Custom
```
                    ┌─────────────────────────┐
                    │ Use Library If Exists   │
                    │ (cuBLAS, CUB, Thrust)   │
                    └───────────┬─────────────┘
                                │ Only if needed
                                ▼
                    ┌─────────────────────────┐
                    │ Write Custom Kernel     │
                    │ (Fusion, special cases) │
                    └─────────────────────────┘
```

### Fusion Decision Tree
```
Multiple kernels? ──► Can they share data in registers/smem? ──► FUSE
                                        │
                                        ▼ No
                                   Keep separate
```

## Success Criteria

After Phase 3, you can:
- [ ] Use warp shuffles for efficient reductions
- [ ] Select appropriate CUDA libraries
- [ ] Identify fusion opportunities
- [ ] Manage memory efficiently at scale
- [ ] Write production-quality CUDA code

## Directory Structure

```
phase3/
├── week13/          # Warp-Level Programming
├── week14/          # CUDA Libraries
├── week15/          # Kernel Fusion
└── week16/          # Memory Management
```
