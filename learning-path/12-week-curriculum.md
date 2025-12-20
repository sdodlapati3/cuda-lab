# CUDA Learning: 15-Week Comprehensive Curriculum

> 🎯 **Goal:** Become proficient in CUDA programming in 15 focused weeks  
> ⏱️ **Time commitment:** 4-6 hours per day, 5-6 days per week  
> 📅 **Total:** ~375 hours of focused learning

---

## 📜 Learning Philosophy

> **CUDA C++ First, Python/Numba as Optional Backup**

All notebooks in this curriculum follow this structure:
1. **CUDA C++ code examples** - The PRIMARY learning material
2. **Python/Numba code** - OPTIONAL alternative for quick Colab testing
3. **Exercises** - Write CUDA C++ first, Python optional

**Why CUDA C++ first?**
- Industry standard for GPU programming
- Transferable skills to any CUDA project
- Deep understanding of memory management, synchronization
- Production CUDA code is written in C++

**Day 5 each week:** Practice & Quiz (consolidation day, no new content)

---

## Overview

This is a **comprehensive, achievable** curriculum that focuses on practical skills through 14 weeks. Master these weeks and you'll be able to:

- Write efficient CUDA kernels from scratch
- Optimize GPU code using profiling tools
- Handle real-world problems (image processing, matrix operations, ML primitives)
- Understand multi-GPU and advanced patterns
- **Leverage Tensor Cores for AI/ML workloads**
- **Build production-ready CUDA applications**

---

## 📊 Progress Tracker

| Week | Focus | Status | Completed |
|------|-------|--------|-----------|
| 1 | GPU Fundamentals | ✅ Complete | 2025-01 |
| 2 | Memory Patterns | 🔄 In Progress | |
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
| **13** | **Tensor Cores & Mixed Precision** | ⬜ Not Started | |
| **14** | **Real-World Applications** | ⬜ Not Started | |
| **15** | **Dynamic Parallelism (CDP)** | ⬜ Not Started | |

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

### Daily Schedule
| Day | Topic | Materials |
|-----|-------|-----------|
| 1 | Memory coalescing | [day-1-memory-coalescing.ipynb](week-02/day-1-memory-coalescing.ipynb) |
| 2 | Shared memory | [day-2-shared-memory.ipynb](week-02/day-2-shared-memory.ipynb) |
| 3 | Bank conflicts | [day-3-bank-conflicts.ipynb](week-02/day-3-bank-conflicts.ipynb) |
| 4 | Constant & texture memory | [day-4-special-memory.ipynb](week-02/day-4-special-memory.ipynb) |
| 5 | Practice & Quiz | Exercises + [checkpoint-quiz.md](week-02/checkpoint-quiz.md) |

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

## Week 13: Tensor Cores & Mixed Precision 🆕

### Learning Goals
- Understand Tensor Core architecture
- Program with WMMA API
- Implement mixed precision training concepts
- Use cuBLAS with Tensor Core acceleration

### Daily Schedule
| Day | Topic | Materials |
|-----|-------|-----------|
| 1 | Tensor Core basics | [day-1-tensor-core-basics.ipynb](week-13/day-1-tensor-core-basics.ipynb) |
| 2 | WMMA programming | [day-2-wmma.ipynb](week-13/day-2-wmma.ipynb) |
| 3 | Mixed precision | [day-3-mixed-precision.ipynb](week-13/day-3-mixed-precision.ipynb) |
| 4 | cuBLAS Tensor Cores | [day-4-cublas-tensor.ipynb](week-13/day-4-cublas-tensor.ipynb) |
| 5 | Practice & Quiz | Exercises + [checkpoint-quiz.md](week-13/checkpoint-quiz.md) |

### Topics
1. Tensor Core vs CUDA Core architecture
2. WMMA fragment types (matrix_a, matrix_b, accumulator)
3. FP16, TF32, BF16 data types
4. Loss scaling for gradient underflow prevention
5. cuBLAS math modes and compute types

### Hardware Requirements
- GPU with Compute Capability 7.0+ (Volta, Turing, Ampere, Hopper)
- TF32 and BF16 require Ampere (SM 8.0+)

### Deliverables
- [ ] Implement basic WMMA matrix multiply
- [ ] Compare performance: CUDA cores vs Tensor Cores
- [ ] Implement loss scaling for mixed precision
- [ ] Quiz score ≥ 24/30

---

## Week 14: Real-World CUDA Applications 🆕

### Learning Goals
- Implement fused kernels for production
- Optimize attention mechanisms
- Build PyTorch CUDA extensions
- Benchmark professionally

### Daily Schedule
| Day | Topic | Materials |
|-----|-------|-----------|
| 1 | Fused kernels | [day-1-fused-kernels.ipynb](week-14/day-1-fused-kernels.ipynb) |
| 2 | Attention mechanisms | [day-2-attention.ipynb](week-14/day-2-attention.ipynb) |
| 3 | PyTorch extensions | [day-3-pytorch-extensions.ipynb](week-14/day-3-pytorch-extensions.ipynb) |
| 4 | Benchmarking | [day-4-benchmarking.ipynb](week-14/day-4-benchmarking.ipynb) |
| 5 | Practice & Quiz | Exercises + [checkpoint-quiz.md](week-14/checkpoint-quiz.md) |

### Topics
1. Kernel fusion patterns (softmax, layernorm)
2. Memory-efficient attention (tiling, Flash Attention concepts)
3. PyTorch C++ extension API
4. autograd.Function integration
5. Professional benchmarking methodology

### Deliverables
- [ ] Implement fused softmax kernel
- [ ] Build basic PyTorch CUDA extension
- [ ] Benchmark with statistical analysis
- [ ] Quiz score ≥ 24/30

---

## Week 15: Dynamic Parallelism (CDP) 🆕

### Learning Goals
- Understand CUDA Dynamic Parallelism (CDP/CDP2)
- Launch kernels from within device code
- Implement recursive algorithms on GPU
- Optimize CDP workloads

### Daily Schedule
| Day | Topic | Materials |
|-----|-------|-----------|
| 1 | CDP fundamentals | [day-1-cdp-fundamentals.ipynb](week-15/day-1-cdp-fundamentals.ipynb) |
| 2 | Recursive algorithms | [day-2-recursive-algorithms.ipynb](week-15/day-2-recursive-algorithms.ipynb) |
| 3 | Adaptive algorithms | [day-3-adaptive-algorithms.ipynb](week-15/day-3-adaptive-algorithms.ipynb) |
| 4 | CDP optimization | [day-4-cdp-optimization.ipynb](week-15/day-4-cdp-optimization.ipynb) |
| 5 | Practice & Quiz | Exercises + [checkpoint-quiz.md](week-15/checkpoint-quiz.md) |

### Topics
1. Parent-child kernel relationship
2. Memory visibility and coherence
3. Device-side streams and synchronization
4. Recursion depth limits
5. Tail launch optimization (CDP2)

### Hardware Requirements
- GPU with Compute Capability 3.5+ (basic CDP)
- GPU with Compute Capability 7.0+ (CDP2 tail launch)

### Compilation
```bash
nvcc -rdc=true program.cu -o program -lcudadevrt
```

### Deliverables
- [ ] Implement GPU quicksort with CDP
- [ ] Build adaptive grid algorithm
- [ ] Compare CDP vs iterative approaches
- [ ] Quiz score ≥ 24/30

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
- [ ] **Program Tensor Cores with WMMA for AI workloads**
- [ ] **Implement mixed precision training techniques**
- [ ] **Build custom PyTorch CUDA extensions**
- [ ] **Benchmark GPU code professionally**
- [ ] **Implement dynamic parallelism patterns**

---

*This 15-week curriculum provides comprehensive CUDA training from fundamentals to production-ready applications.*
