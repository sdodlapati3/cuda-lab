# CUDA Learning: 12-Week MVP Curriculum

> 🎯 **Goal:** Become proficient in CUDA programming in 12 focused weeks  
> ⏱️ **Time commitment:** 4-6 hours per day, 5-6 days per week  
> 📅 **Total:** ~300 hours of focused learning

---

## Overview

This is a **streamlined, achievable** curriculum that focuses on practical skills over comprehensive coverage. Master these 12 weeks and you'll be able to:

- Write efficient CUDA kernels from scratch
- Optimize GPU code using profiling tools
- Handle real-world problems (image processing, matrix operations, ML primitives)
- Understand multi-GPU and advanced patterns

---

## 📊 Progress Tracker

| Week | Focus | Status | Completed |
|------|-------|--------|-----------|
| 1 | GPU Fundamentals | ⬜ Not Started | |
| 2 | Memory Patterns | ⬜ Not Started | |
| 3 | Parallel Patterns I | ⬜ Not Started | |
| 4 | Reduction & Atomics | ⬜ Not Started | |
| 5 | Prefix Sum (Scan) | ⬜ Not Started | |
| 6 | Matrix Operations | ⬜ Not Started | |
| 7 | Memory Optimization | ⬜ Not Started | |
| 8 | Profiling & Analysis | ⬜ Not Started | |
| 9 | Streams & Concurrency | ⬜ Not Started | |
| 10 | Advanced Patterns | ⬜ Not Started | |
| 11 | Multi-GPU & Scaling | ⬜ Not Started | |
| 12 | Capstone Project | ⬜ Not Started | |

---

## Week 1: GPU Fundamentals

### Learning Goals
- Understand CPU vs GPU architecture
- Write and launch basic CUDA kernels
- Master thread indexing (1D, 2D, 3D)
- Handle memory transfers

### Daily Schedule
| Day | Topic | Materials |
|-----|-------|-----------|
| 1 | GPU basics, device query | [day-1-gpu-basics.ipynb](week-01/day-1-gpu-basics.ipynb) |
| 2 | Thread indexing | [day-2-thread-indexing.ipynb](week-01/day-2-thread-indexing.ipynb) |
| 3 | Memory management | [day-3-memory-basics.ipynb](week-01/day-3-memory-basics.ipynb) |
| 4 | Error handling & debugging | [day-4-error-handling.ipynb](week-01/day-4-error-handling.ipynb) |
| 5 | Practice & Quiz | Exercises + [checkpoint-quiz.md](week-01/checkpoint-quiz.md) |

### Deliverables
- [ ] All notebook exercises completed
- [ ] Quiz score ≥ 25/30
- [ ] Can write a kernel from scratch without reference

---

## Week 2: Memory Patterns & Optimization

### Learning Goals
- Understand coalesced memory access
- Use shared memory effectively
- Avoid bank conflicts
- Profile memory behavior

### Topics
1. Global memory access patterns
2. Memory coalescing rules
3. Shared memory introduction
4. Shared memory bank conflicts
5. Constant memory and texture memory

### Project
**Image Filter:** Implement Gaussian blur using shared memory tiling

---

## Week 3: Parallel Patterns I (Vector Operations)

### Learning Goals
- Implement grid-stride loops professionally
- Handle edge cases and arbitrary sizes
- Vector operations at scale
- Fused operations for better performance

### Topics
1. Vector addition/subtraction/multiplication
2. Vector dot product (setup for reduction)
3. SAXPY and BLAS-like operations
4. Multiple operations per thread
5. Fusing operations to reduce memory traffic

### Project
**Vector Math Library:** Complete library with add, sub, mul, div, dot, norm, scale

---

## Week 4: Reduction & Atomics

### Learning Goals
- Implement tree reduction
- Understand warp-level primitives
- Use atomic operations correctly
- Combine reduction techniques

### Topics
1. Naive reduction (with divergence)
2. Sequential addressing (no divergence)
3. Warp shuffle reduction
4. Atomic operations
5. Combining techniques for optimal reduction

### Project
**Statistical Functions:** sum, mean, min, max, variance implemented on GPU

---

## Week 5: Prefix Sum (Scan)

### Learning Goals
- Understand scan as fundamental parallel primitive
- Implement work-efficient scan
- Handle arrays larger than block size
- Apply scan to real problems

### Topics
1. Inclusive vs exclusive scan
2. Hillis-Steele algorithm
3. Blelloch algorithm
4. Large array scan with multiple blocks
5. Applications: stream compaction, radix sort

### Project
**Stream Compaction:** Filter array to keep only positive values

---

## Week 6: Matrix Operations

### Learning Goals
- Implement optimized matrix multiply
- Understand tiling strategies
- Achieve good performance relative to cuBLAS
- Matrix transpose optimization

### Topics
1. Naive matrix multiply
2. Tiled matrix multiply with shared memory
3. Rectangular matrix handling
4. Matrix transpose (naive vs coalesced)
5. Comparison with cuBLAS

### Project
**Matrix Library:** multiply, transpose, add with 80%+ cuBLAS performance

---

## Week 7: Memory Optimization Deep Dive

### Learning Goals
- Master occupancy analysis
- Reduce register pressure
- Optimize shared memory usage
- Understand L1/L2 cache behavior

### Topics
1. Occupancy calculator usage
2. Register spilling and local memory
3. Shared memory configurations
4. Cache optimization strategies
5. Unified memory and prefetching

### Project
Optimize Week 6 matrix multiply to 95%+ cuBLAS performance

---

## Week 8: Profiling & Analysis

### Learning Goals
- Master Nsight Compute
- Use Nsight Systems for timeline analysis
- Understand roofline model
- Identify and fix bottlenecks

### Topics
1. Nsight Compute basics
2. Key metrics: bandwidth, occupancy, instruction mix
3. Nsight Systems for system-wide analysis
4. Roofline analysis
5. Systematic optimization workflow

### Project
Profile and optimize a provided slow kernel by 10x

---

## Week 9: Streams & Concurrency

### Learning Goals
- Overlap computation and transfers
- Use multiple streams effectively
- Understand CUDA events
- Implement async patterns

### Topics
1. CUDA streams introduction
2. Async memory operations
3. Overlap patterns (H2D, kernel, D2H)
4. Events for timing and synchronization
5. Multi-stream best practices

### Project
**Pipeline Processing:** Overlap data loading with processing for image batch

---

## Week 10: Advanced Patterns

### Learning Goals
- Implement histogram efficiently
- Understand sorting on GPU
- Use CUB and Thrust effectively
- Cooperative groups basics

### Topics
1. Histogram (atomic, privatization)
2. Sorting (odd-even, bitonic, radix)
3. CUB library for common operations
4. Thrust for STL-like GPU operations
5. Cooperative groups for flexible sync

### Project
**Data Analysis Pipeline:** Load data, histogram, sort, statistics

---

## Week 11: Multi-GPU & Scaling

### Learning Goals
- Work with multiple GPUs
- Understand peer-to-peer access
- Distribute work across devices
- Intro to NCCL

### Topics
1. Device enumeration and management
2. P2P memory access
3. Work distribution strategies
4. Multi-GPU reduction
5. NCCL collective operations

### Project
**Multi-GPU Matrix Multiply:** Distribute large matrix across 2+ GPUs

---

## Week 12: Capstone Project

### Goals
Apply everything learned to a complete, optimized application.

### Choose One:
1. **ML Inference Engine**
   - Custom matrix multiply
   - Fused activation functions
   - Batch processing with streams

2. **Image Processing Pipeline**
   - Multi-filter convolution
   - Edge detection
   - Color space conversions
   - Batch processing

3. **Scientific Simulation**
   - N-body simulation
   - Particle system
   - Heat diffusion

### Requirements
- Optimized (profile-guided)
- Multi-GPU or heavily streamed
- Well-documented
- Performance comparison with CPU/library

---

## 📚 Resources

### Primary Materials
- Interactive notebooks in `learning-path/week-XX/`
- Practice exercises in `practice/`
- Reference docs in `cuda-programming-guide/`

### Supplementary
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [CUB Documentation](https://nvlabs.github.io/cub/)

---

## 🎯 Success Criteria

By completing this curriculum, you should be able to:

- [ ] Write CUDA kernels from scratch without reference
- [ ] Identify memory vs compute bottlenecks
- [ ] Optimize kernels using profiling data
- [ ] Implement common parallel patterns
- [ ] Work with multi-GPU systems
- [ ] Achieve 80%+ theoretical peak on bandwidth-bound kernels
- [ ] Achieve 50%+ theoretical peak on compute-bound kernels

---

*This curriculum replaces the original 26-week plan with a more focused, achievable path.*
