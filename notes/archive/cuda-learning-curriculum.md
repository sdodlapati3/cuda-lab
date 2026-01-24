# CUDA Learning Curriculum

> 🎯 **Goal:** Master CUDA programming from beginner to advanced  
> ⏱️ **Commitment:** 15 hours/day × 6 months (~2,700 hours)  
> 📅 **Start Date:** December 2025

---

## 📊 Progress Tracker

| Phase | Duration | Status | Start | End |
|-------|----------|--------|-------|-----|
| Phase 1: Foundations | 3 weeks | ⬜ Not Started | | |
| Phase 2: Core CUDA | 4 weeks | ⬜ Not Started | | |
| Phase 3: Memory Mastery | 3 weeks | ⬜ Not Started | | |
| Phase 4: Performance | 4 weeks | ⬜ Not Started | | |
| Phase 5: Advanced | 4 weeks | ⬜ Not Started | | |
| Phase 6: Specialization | 4 weeks | ⬜ Not Started | | |
| Phase 7: Capstone | 4 weeks | ⬜ Not Started | | |

---

## Phase 1: Foundations (Weeks 1-3)

### Week 1: GPU Architecture & Setup
**Goal:** Understand why GPUs exist and how they differ from CPUs

#### Tutorials to Create
- [ ] `tutorials/01-foundations/01-cpu-vs-gpu.md` - Why GPUs? Parallelism concepts
- [ ] `tutorials/01-foundations/02-gpu-architecture.md` - SMs, cores, warps, threads
- [ ] `tutorials/01-foundations/03-cuda-ecosystem.md` - Toolkit, driver, runtime
- [ ] `tutorials/01-foundations/04-development-setup.md` - nvcc, nsight, debugging

#### Exercises
```
practice/01-foundations/
├── ex01-device-query/        # Query GPU properties
├── ex02-hello-gpu/           # First __global__ function
├── ex03-thread-indexing/     # Print thread/block IDs
└── ex04-error-handling/      # Proper CUDA error checks
```

#### Reading
- [x] cuda-programming-guide/01-introduction/*.md
- [ ] NVIDIA GPU Architecture Whitepaper (your GPU generation)

---

### Week 2: CUDA Programming Model
**Goal:** Understand threads, blocks, grids, and how work is organized

#### Tutorials to Create
- [ ] `tutorials/01-foundations/05-execution-model.md` - Kernels, threads, blocks
- [ ] `tutorials/01-foundations/06-thread-hierarchy.md` - 1D, 2D, 3D indexing
- [ ] `tutorials/01-foundations/07-kernel-launch.md` - <<<grid, block>>> syntax
- [ ] `tutorials/01-foundations/08-synchronization-basics.md` - __syncthreads()

#### Exercises
```
practice/01-foundations/
├── ex05-1d-indexing/         # Vector operations
├── ex06-2d-indexing/         # Matrix element access
├── ex07-grid-stride-loop/    # Handle arbitrary sizes
├── ex08-block-sync/          # Synchronization within blocks
└── ex09-launch-config/       # Experiment with different configs
```

#### Mini-Project
**Vector Calculator:** Add, subtract, multiply, divide large vectors
- Handle arbitrary sizes
- Compare CPU vs GPU performance
- Profile with nvprof/nsight

---

### Week 3: Memory Fundamentals
**Goal:** Understand basic memory operations and host-device transfers

#### Tutorials to Create
- [ ] `tutorials/01-foundations/09-memory-spaces.md` - Global, shared, local, constant
- [ ] `tutorials/01-foundations/10-memory-allocation.md` - cudaMalloc, cudaFree
- [ ] `tutorials/01-foundations/11-data-transfer.md` - cudaMemcpy patterns
- [ ] `tutorials/01-foundations/12-pinned-memory.md` - cudaMallocHost benefits

#### Exercises
```
practice/01-foundations/
├── ex10-malloc-memcpy/       # Basic allocation & transfer
├── ex11-pinned-vs-pageable/  # Compare transfer speeds
├── ex12-2d-memory/           # cudaMallocPitch, cudaMemcpy2D
└── ex13-unified-memory-intro/# cudaMallocManaged basics
```

#### Mini-Project
**Image Loader:** Load image to GPU, apply simple filter (invert, grayscale)

---

## Phase 2: Core CUDA Programming (Weeks 4-7)

### Week 4: Parallel Patterns - Reduction
**Goal:** Master the fundamental reduction pattern

#### Tutorials to Create
- [ ] `tutorials/02-core/01-reduction-naive.md` - Sum of array (naive)
- [ ] `tutorials/02-core/02-reduction-optimized.md` - Sequential addressing
- [ ] `tutorials/02-core/03-reduction-warp.md` - Warp-level reduction
- [ ] `tutorials/02-core/04-reduction-atomic.md` - Using atomics

#### Exercises
```
practice/02-core/reduction/
├── ex01-naive-sum/           # Interleaved addressing
├── ex02-sequential-sum/      # Sequential addressing (no divergence)
├── ex03-first-add/           # First add during load
├── ex04-unroll-last-warp/    # Unroll last warp
├── ex05-complete-unroll/     # Fully unrolled
├── ex06-multi-add/           # Multiple elements per thread
├── ex07-warp-shuffle/        # Using __shfl_down_sync
└── ex08-cub-reduction/       # Compare with CUB library
```

#### Benchmark Project
Compare all reduction implementations, create performance chart

---

### Week 5: Parallel Patterns - Scan (Prefix Sum)
**Goal:** Master inclusive and exclusive scan

#### Tutorials to Create
- [ ] `tutorials/02-core/05-scan-concepts.md` - Inclusive vs exclusive
- [ ] `tutorials/02-core/06-scan-hillis-steele.md` - Work-efficient scan
- [ ] `tutorials/02-core/07-scan-blelloch.md` - Blelloch scan
- [ ] `tutorials/02-core/08-scan-large-arrays.md` - Multi-block scan

#### Exercises
```
practice/02-core/scan/
├── ex01-inclusive-naive/     # Basic inclusive scan
├── ex02-exclusive-naive/     # Basic exclusive scan
├── ex03-work-efficient/      # Blelloch algorithm
├── ex04-bank-conflicts/      # Avoid shared memory bank conflicts
├── ex05-large-array/         # Scan arrays > block size
└── ex06-stream-compaction/   # Application: compact array
```

---

### Week 6: Parallel Patterns - Histogram & Sorting
**Goal:** Learn histogram and parallel sorting algorithms

#### Tutorials to Create
- [ ] `tutorials/02-core/09-histogram-atomic.md` - Atomic histogram
- [ ] `tutorials/02-core/10-histogram-privatization.md` - Per-thread histograms
- [ ] `tutorials/02-core/11-radix-sort.md` - Radix sort on GPU
- [ ] `tutorials/02-core/12-merge-sort.md` - Parallel merge sort

#### Exercises
```
practice/02-core/histogram/
├── ex01-atomic-global/       # Global memory atomics
├── ex02-atomic-shared/       # Shared memory atomics
├── ex03-privatization/       # Per-thread histograms
└── ex04-aggregation/         # Combine approaches

practice/02-core/sorting/
├── ex01-odd-even/            # Odd-even transposition
├── ex02-bitonic/             # Bitonic sort
├── ex03-radix/               # Radix sort
└── ex04-thrust-sort/         # Compare with Thrust
```

---

### Week 7: Matrix Operations
**Goal:** Implement optimized matrix operations

#### Tutorials to Create
- [ ] `tutorials/02-core/13-matrix-add.md` - 2D grid for matrices
- [ ] `tutorials/02-core/14-matrix-mul-naive.md` - Naive matrix multiply
- [ ] `tutorials/02-core/15-matrix-mul-tiled.md` - Shared memory tiling
- [ ] `tutorials/02-core/16-matrix-transpose.md` - Coalesced transpose

#### Exercises
```
practice/02-core/matrix/
├── ex01-matrix-add/          # Element-wise operations
├── ex02-matrix-mul-naive/    # Naive O(n³)
├── ex03-matrix-mul-tiled/    # Tiled with shared memory
├── ex04-matrix-mul-rect/     # Non-square matrices
├── ex05-transpose-naive/     # Naive transpose
├── ex06-transpose-coalesced/ # Coalesced transpose
├── ex07-transpose-conflict/  # Avoid bank conflicts
└── ex08-cublas-comparison/   # Compare with cuBLAS
```

#### Major Project
**Matrix Library:** Complete matrix library with add, multiply, transpose, inverse

---

## Phase 3: Memory Mastery (Weeks 8-10)

### Week 8: Memory Hierarchy Deep Dive
**Goal:** Master all memory types and their optimal use cases

#### Tutorials to Create
- [ ] `tutorials/03-memory/01-memory-hierarchy.md` - Complete overview
- [ ] `tutorials/03-memory/02-global-memory.md` - Coalescing patterns
- [ ] `tutorials/03-memory/03-shared-memory.md` - Bank conflicts, patterns
- [ ] `tutorials/03-memory/04-constant-memory.md` - Broadcast reads
- [ ] `tutorials/03-memory/05-texture-memory.md` - Spatial locality
- [ ] `tutorials/03-memory/06-registers.md` - Register pressure

#### Exercises
```
practice/03-memory/
├── ex01-coalescing-patterns/ # Test different access patterns
├── ex02-shared-tiling/       # Tiling strategies
├── ex03-bank-conflicts/      # Detect and avoid conflicts
├── ex04-constant-broadcast/  # Constant memory broadcast
├── ex05-texture-interpolation/ # Texture filtering
├── ex06-register-spilling/   # Monitor register usage
└── ex07-occupancy-analysis/  # Use occupancy calculator
```

---

### Week 9: Unified Memory & Advanced Allocation
**Goal:** Master modern memory management

#### Tutorials to Create
- [ ] `tutorials/03-memory/07-unified-memory.md` - cudaMallocManaged
- [ ] `tutorials/03-memory/08-memory-prefetch.md` - cudaMemPrefetchAsync
- [ ] `tutorials/03-memory/09-memory-advise.md` - cudaMemAdvise hints
- [ ] `tutorials/03-memory/10-virtual-memory.md` - Virtual memory management

#### Exercises
```
practice/03-memory/
├── ex08-unified-basics/      # Basic unified memory
├── ex09-prefetch-patterns/   # Prefetching strategies
├── ex10-oversubscription/    # Handle > GPU memory
├── ex11-access-counters/     # Monitor page faults
└── ex12-memory-pools/        # Stream-ordered allocation
```

---

### Week 10: Memory Optimization Project
**Goal:** Apply all memory knowledge to real problems

#### Projects
```
practice/03-memory/projects/
├── stencil-2d/               # 2D stencil with halos
├── sparse-matrix/            # SpMV with various formats
└── image-convolution/        # Optimized convolution
```

---

## Phase 4: Performance Optimization (Weeks 11-14)

### Week 11: Profiling & Analysis
**Goal:** Master NVIDIA profiling tools

#### Tutorials to Create
- [ ] `tutorials/04-performance/01-nsight-compute.md` - Kernel profiling
- [ ] `tutorials/04-performance/02-nsight-systems.md` - System profiling
- [ ] `tutorials/04-performance/03-metrics.md` - Key metrics to watch
- [ ] `tutorials/04-performance/04-roofline.md` - Roofline model analysis

#### Exercises
```
practice/04-performance/profiling/
├── ex01-basic-profiling/     # First profile session
├── ex02-memory-bound/        # Identify memory bottlenecks
├── ex03-compute-bound/       # Identify compute bottlenecks
├── ex04-occupancy-tuning/    # Improve occupancy
└── ex05-roofline/            # Create roofline plots
```

---

### Week 12: Kernel Optimization Techniques
**Goal:** Learn systematic optimization approaches

#### Tutorials to Create
- [ ] `tutorials/04-performance/05-instruction-optimization.md` - ILP, math
- [ ] `tutorials/04-performance/06-launch-config.md` - Optimal grid/block
- [ ] `tutorials/04-performance/07-divergence.md` - Minimize branch divergence
- [ ] `tutorials/04-performance/08-occupancy.md` - Maximize occupancy

#### Exercises
```
practice/04-performance/optimization/
├── ex01-instruction-mix/     # Balance instructions
├── ex02-loop-unrolling/      # Manual unrolling
├── ex03-divergence-patterns/ # Reduce divergence
├── ex04-thread-coarsening/   # More work per thread
└── ex05-persistent-threads/  # Persistent kernel pattern
```

---

### Week 13: Streams & Concurrency
**Goal:** Master asynchronous execution

#### Tutorials to Create
- [ ] `tutorials/04-performance/09-streams-basics.md` - Stream creation & use
- [ ] `tutorials/04-performance/10-async-memcpy.md` - Overlap transfer & compute
- [ ] `tutorials/04-performance/11-events.md` - Timing and synchronization
- [ ] `tutorials/04-performance/12-multi-stream.md` - Multiple stream patterns

#### Exercises
```
practice/04-performance/streams/
├── ex01-default-stream/      # Implicit synchronization
├── ex02-concurrent-streams/  # Parallel execution
├── ex03-overlap-h2d-kernel/  # Overlap transfers
├── ex04-double-buffering/    # Pipeline pattern
├── ex05-callbacks/           # Stream callbacks
└── ex06-stream-priorities/   # Priority scheduling
```

---

### Week 14: CUDA Graphs
**Goal:** Master graph-based execution

#### Tutorials to Create
- [ ] `tutorials/04-performance/13-graphs-intro.md` - Why graphs?
- [ ] `tutorials/04-performance/14-graphs-capture.md` - Stream capture
- [ ] `tutorials/04-performance/15-graphs-explicit.md` - Explicit graphs
- [ ] `tutorials/04-performance/16-graphs-update.md` - Graph updates

#### Exercises
```
practice/04-performance/graphs/
├── ex01-simple-graph/        # First CUDA graph
├── ex02-stream-capture/      # Capture existing code
├── ex03-graph-update/        # Update parameters
├── ex04-graph-pipeline/      # Complex pipelines
└── ex05-graph-vs-streams/    # Performance comparison
```

---

## Phase 5: Advanced CUDA (Weeks 15-18)

### Week 15: Warp-Level Programming
**Goal:** Master warp primitives

#### Tutorials to Create
- [ ] `tutorials/05-advanced/01-warp-primitives.md` - Shuffle, vote, match
- [ ] `tutorials/05-advanced/02-warp-reduction.md` - Warp-level reduction
- [ ] `tutorials/05-advanced/03-warp-scan.md` - Warp-level scan
- [ ] `tutorials/05-advanced/04-cooperative-groups.md` - Flexible groups

#### Exercises
```
practice/05-advanced/warp/
├── ex01-shuffle-broadcast/   # __shfl_sync patterns
├── ex02-shuffle-reduce/      # Warp reduction
├── ex03-ballot-vote/         # Voting functions
├── ex04-match/               # Match functions
├── ex05-cg-partition/        # Cooperative groups
└── ex06-cg-reduce/           # CG reductions
```

---

### Week 16: Multi-GPU Programming
**Goal:** Scale to multiple GPUs

#### Tutorials to Create
- [ ] `tutorials/05-advanced/05-multi-gpu-basics.md` - Device management
- [ ] `tutorials/05-advanced/06-peer-access.md` - P2P memory access
- [ ] `tutorials/05-advanced/07-multi-gpu-patterns.md` - Work distribution
- [ ] `tutorials/05-advanced/08-nccl.md` - NCCL for communication

#### Exercises
```
practice/05-advanced/multi-gpu/
├── ex01-device-enumeration/  # Query multiple GPUs
├── ex02-peer-memcpy/         # P2P transfers
├── ex03-split-work/          # Divide computation
├── ex04-multi-gpu-reduce/    # Multi-GPU reduction
└── ex05-nccl-allreduce/      # NCCL collective
```

---

### Week 17: Dynamic Parallelism
**Goal:** Launch kernels from kernels

#### Tutorials to Create
- [ ] `tutorials/05-advanced/09-dynamic-parallelism.md` - Basics
- [ ] `tutorials/05-advanced/10-dp-patterns.md` - Use cases
- [ ] `tutorials/05-advanced/11-dp-optimization.md` - Performance tips
- [ ] `tutorials/05-advanced/12-dp-recursion.md` - Recursive algorithms

#### Exercises
```
practice/05-advanced/dynamic/
├── ex01-simple-dp/           # First dynamic launch
├── ex02-quicksort/           # Recursive quicksort
├── ex03-tree-traversal/      # Adaptive tree
├── ex04-mandelbrot/          # Adaptive refinement
└── ex05-bvh/                 # Bounding volume hierarchy
```

---

### Week 18: Tensor Cores & Mixed Precision
**Goal:** Use Tensor Cores for ML workloads

#### Tutorials to Create
- [ ] `tutorials/05-advanced/13-tensor-cores.md` - WMMA basics
- [ ] `tutorials/05-advanced/14-mixed-precision.md` - FP16, BF16, TF32
- [ ] `tutorials/05-advanced/15-wmma-gemm.md` - Matrix multiply with WMMA
- [ ] `tutorials/05-advanced/16-cutlass.md` - Using CUTLASS

#### Exercises
```
practice/05-advanced/tensor-cores/
├── ex01-wmma-basics/         # First WMMA kernel
├── ex02-wmma-gemm/           # GEMM implementation
├── ex03-mixed-precision/     # FP16 accumulate FP32
└── ex04-cutlass-gemm/        # CUTLASS template
```

---

## Phase 6: Specialization (Weeks 19-22)

Choose 2-3 tracks based on your interests:

### Track A: Deep Learning
```
practice/06-specialization/deep-learning/
├── convolution/              # Conv2D, depthwise, grouped
├── attention/                # Scaled dot-product attention
├── normalization/            # BatchNorm, LayerNorm
├── activation/               # ReLU, GELU, SiLU
├── pooling/                  # Max, average, adaptive
├── embedding/                # Lookup, positional encoding
└── custom-autograd/          # PyTorch/JAX integration
```

### Track B: Scientific Computing
```
practice/06-specialization/scientific/
├── fft/                      # cuFFT, custom FFT
├── linear-algebra/           # LU, QR, SVD, eigensolvers
├── pde-solvers/              # Finite difference, FEM
├── monte-carlo/              # cuRAND, parallel RNG
├── n-body/                   # Gravitational simulation
└── molecular-dynamics/       # Particle simulations
```

### Track C: Computer Vision & Graphics
```
practice/06-specialization/vision/
├── image-processing/         # Filters, morphology
├── feature-detection/        # SIFT, ORB on GPU
├── stereo-matching/          # Disparity computation
├── ray-tracing/              # Path tracing, BVH
├── volume-rendering/         # Medical imaging
└── point-cloud/              # LiDAR processing
```

### Track D: Systems & Infrastructure
```
practice/06-specialization/systems/
├── memory-allocators/        # Custom allocators
├── kernel-fusion/            # JIT compilation
├── scheduling/               # Work stealing, queues
├── compression/              # GPU compression
├── database-ops/             # Hash joins, aggregation
└── networking/               # GPUDirect RDMA
```

---

## Phase 7: Capstone Projects (Weeks 23-26)

### Project 1: End-to-End ML Inference Engine
Build a complete inference engine from scratch:
- Custom memory pool
- Kernel fusion
- Graph optimization
- Multi-batch scheduling
- Benchmark against TensorRT

### Project 2: Scientific Computing Application
Choose one:
- Fluid dynamics simulator (Navier-Stokes)
- Molecular dynamics with CUDA
- Weather simulation
- Quantum computing simulator

### Project 3: Open Source Contribution
- Contribute to CuPy, RAPIDS, or similar
- Create a CUDA library for a specific domain
- Write performance benchmarks and blog posts

---

## 📚 Supplementary Resources

### Books
1. "CUDA by Example" - Sanders & Kandrot (Beginner)
2. "Programming Massively Parallel Processors" - Kirk & Hwu (Intermediate)
3. "CUDA Handbook" - Nicholas Wilt (Advanced reference)

### Online Courses
1. NVIDIA DLI - Fundamentals of Accelerated Computing
2. Coursera - GPU Programming Specialization
3. Udacity - Intro to Parallel Programming

### Documentation
- [CUDA Programming Guide](../cuda-programming-guide/index.md) ← Already downloaded!
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)

### Code References
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [CUB Library](https://github.com/NVIDIA/cub)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [Thrust](https://github.com/NVIDIA/thrust)

---

## 📅 Weekly Schedule Template

| Time | Activity |
|------|----------|
| 06:00-09:00 | Tutorial reading & notes |
| 09:00-09:30 | Break |
| 09:30-12:30 | Coding exercises |
| 12:30-13:30 | Lunch |
| 13:30-16:30 | Project work |
| 16:30-17:00 | Break |
| 17:00-19:00 | Profiling & optimization |
| 19:00-20:00 | Dinner |
| 20:00-21:00 | Review & documentation |

---

## ✅ Milestones

- [ ] **Week 3:** Complete first GPU project (vector calculator)
- [ ] **Week 7:** Implement optimized matrix multiply (>1 TFLOPS)
- [ ] **Week 10:** Complete memory-optimized stencil code
- [ ] **Week 14:** Create pipeline with CUDA graphs
- [ ] **Week 18:** Working Tensor Core GEMM
- [ ] **Week 22:** Specialization track complete
- [ ] **Week 26:** Capstone project deployed

---

*Last updated: December 2025*
