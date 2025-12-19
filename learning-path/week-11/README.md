# Week 11: Cooperative Groups & Dynamic Parallelism

## Overview
This week covers advanced thread cooperation patterns and the ability to launch kernels from within kernels.

## Topics
- **Day 1**: Cooperative Groups Basics - Thread groups and synchronization
- **Day 2**: Grid-Wide Synchronization - Full grid cooperation
- **Day 3**: Dynamic Parallelism Basics - Launching kernels from GPU
- **Day 4**: Nested Kernel Patterns - Practical dynamic parallelism
- **Day 5**: Practice & Quiz

## Learning Objectives
By the end of this week, you will:
- Understand cooperative group hierarchies
- Synchronize across thread blocks
- Launch child kernels from parent kernels
- Apply dynamic parallelism to recursive problems

## Prerequisites
- Weeks 1-10 completed
- Strong understanding of CUDA memory model
- Familiarity with CUDA graphs

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Cooperative Groups | Thread block groups, tile groups, coalesced groups |
| 2 | Grid-Wide Sync | cooperative_groups::grid_group, launch restrictions |
| 3 | Dynamic Parallelism | cudaLaunchDevice, device runtime, streams in kernels |
| 4 | Nested Patterns | Recursive kernels, work decomposition, synchronization |
| 5 | Practice & Quiz | Exercises and checkpoint assessment |

## Key APIs

```cpp
// Cooperative Groups
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

cg::thread_block tb = cg::this_thread_block();
cg::thread_block_tile<32> tile = cg::tiled_partition<32>(tb);
cg::grid_group grid = cg::this_grid();

// Dynamic Parallelism
__global__ void parent_kernel() {
    child_kernel<<<blocks, threads>>>();
    cudaDeviceSynchronize();  // Device-side sync
}
```

## Resources
- [CUDA Programming Guide - Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
- [CUDA Programming Guide - Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dynamic-parallelism)
