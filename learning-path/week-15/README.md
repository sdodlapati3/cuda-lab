# Week 15: Dynamic Parallelism (CDP2)

## Overview
This week explores CUDA Dynamic Parallelism - the ability for GPU kernels to launch other kernels directly from the device. This enables recursive algorithms, adaptive workloads, and hierarchical parallelism patterns.

## Learning Objectives
By the end of this week, you will:
- Understand parent-child kernel relationships
- Implement recursive algorithms on the GPU
- Use device-side memory management
- Apply CDP2 optimizations (tail launch)
- Build adaptive algorithms with dynamic grid sizing

## Prerequisites
- Weeks 1-14 completion
- Solid understanding of kernel launches and memory management
- Familiarity with recursive algorithm concepts

## Daily Schedule

### Day 1: CDP Fundamentals
- Parent and child kernel concepts
- Device-side kernel launch syntax
- Memory visibility between grids
- Device synchronization

### Day 2: Recursive Algorithms
- GPU quicksort implementation
- Tree traversal patterns
- Recursion depth considerations
- Stack memory management

### Day 3: Adaptive Algorithms
- Dynamic grid sizing
- Workload-dependent parallelism
- Adaptive mesh refinement concepts
- Octree/quadtree construction

### Day 4: CDP Optimization
- CDP2 tail launch optimization
- Minimizing launch overhead
- Memory efficiency patterns
- When to use (and avoid) CDP

## Key Concepts

### Parent-Child Relationship
```cpp
__global__ void parentKernel(int* data, int n) {
    if (n > THRESHOLD) {
        // Launch child kernel from GPU
        childKernel<<<gridDim, blockDim>>>(data, n/2);
        cudaDeviceSynchronize();  // Wait for children
    }
}
```

### CDP2 Improvements (CUDA 12+)
- Tail launch: Child inherits parent's resources
- Reduced memory overhead
- Better nested parallelism efficiency

## Hardware Requirements
- Compute Capability 3.5+ (basic CDP)
- Compute Capability 7.0+ (recommended for CDP2)
- Compile with: `nvcc -rdc=true -lcudadevrt`

## Resources
- [CUDA Programming Guide - Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-dynamic-parallelism)
- [CDP Best Practices](https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/)

## Exercises
Complete the checkpoint quiz after finishing all notebooks.

## Time Estimate
~25 hours total
- Notebooks: 16 hours
- Practice: 6 hours
- Quiz: 3 hours
