# Week 12: Multi-GPU & Advanced Topics

## Overview
This final week covers multi-GPU programming and brings together advanced concepts for a capstone project.

## Topics
- **Day 1**: Multi-GPU Basics - Device management and peer access
- **Day 2**: Multi-GPU Patterns - Data distribution and communication
- **Day 3**: Advanced Optimization Review - Comprehensive optimization strategies
- **Day 4**: Capstone Project - Complete application development
- **Day 5**: Review & Next Steps

## Learning Objectives
By the end of this week, you will:
- Program applications using multiple GPUs
- Implement efficient data distribution patterns
- Apply advanced optimization strategies
- Build a complete CUDA application from scratch

## Prerequisites
- Weeks 1-11 completed
- Strong understanding of all CUDA concepts covered

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Multi-GPU Basics | cudaSetDevice, peer access, memory copy |
| 2 | Multi-GPU Patterns | Domain decomposition, halo exchange, load balancing |
| 3 | Optimization Review | Memory, compute, latency optimization |
| 4 | Capstone Project | Full application development |
| 5 | Review & Next Steps | Portfolio review, continuing education |

## Key APIs

```cpp
// Device Management
cudaGetDeviceCount(&count);
cudaSetDevice(deviceId);

// Peer Access
cudaDeviceCanAccessPeer(&canAccess, device, peerDevice);
cudaDeviceEnablePeerAccess(peerDevice, 0);
cudaMemcpyPeer(dst, dstDevice, src, srcDevice, size);

// Unified Memory for Multi-GPU
cudaMallocManaged(&ptr, size);
cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
cudaMemPrefetchAsync(ptr, size, device, stream);
```

## Resources
- [CUDA Programming Guide - Multi-GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-gpu)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
