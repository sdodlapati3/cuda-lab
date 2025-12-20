# Week 16: Virtual Memory Management & Advanced Multi-GPU

## Overview
This advanced week covers Virtual Memory Management (VMM) APIs and sophisticated multi-GPU programming patterns. These are production-level topics used in large-scale CUDA applications.

## Learning Objectives
By the end of this week, you will be able to:
- Use VMM APIs for explicit memory control
- Build growable GPU data structures
- Implement stream-ordered memory allocation
- Configure multi-GPU peer access with VMM
- Build custom memory allocators

## Prerequisites
- Weeks 1-15 completion
- Understanding of CUDA streams and async operations
- Familiarity with Driver API concepts

## Daily Schedule

| Day | Topic | Focus |
|-----|-------|-------|
| 1 | VMM Fundamentals | Address reservation, physical allocation, mapping |
| 2 | Stream-Ordered Memory | `cudaMallocAsync`, memory pools |
| 3 | Multi-GPU VMM | Peer access, fabric handles, NVLink |
| 4 | Custom Allocators | Building efficient memory managers |
| 5 | Practice & Quiz | Exercises + Assessment |

## Key Concepts

### Virtual Memory Management
```cpp
// Separate virtual address and physical memory
CUdeviceptr ptr;
cuMemAddressReserve(&ptr, reserveSize, 0, 0, 0);  // Reserve VA
cuMemCreate(&handle, physSize, &prop, 0);          // Create physical
cuMemMap(ptr, physSize, 0, handle, 0);             // Map phys→virt
cuMemSetAccess(ptr, physSize, &access, 1);         // Set permissions
```

### Stream-Ordered Allocation
```cpp
// Memory tied to stream ordering
cudaMallocAsync(&ptr, size, stream);  // Allocate
kernel<<<grid, block, 0, stream>>>(ptr);
cudaFreeAsync(ptr, stream);            // No sync needed!
```

## Hardware Requirements
- GPU with Compute Capability 6.0+ (VMM)
- Multi-GPU system for Day 3 (optional)
- NVLink for optimal performance (optional)

## Key APIs
- `cuMemAddressReserve` / `cuMemAddressFree`
- `cuMemCreate` / `cuMemRelease`
- `cuMemMap` / `cuMemUnmap`
- `cuMemSetAccess`
- `cudaMallocAsync` / `cudaFreeAsync`
- `cudaMemPool*` functions

## Time Commitment
- Study: 4-5 hours/day
- Practice: 2-3 hours/day
- Total: ~25 hours
