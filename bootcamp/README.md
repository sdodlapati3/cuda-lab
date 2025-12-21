# CUDA Mastery Bootcamp: 12-Month Intensive

> **Target Audience:** ML researchers and engineers committed to becoming GPU performance experts.
> 
> **Prerequisite:** Complete the [18-Week CUDA Tutorial](../learning-path/README.md) OR equivalent experience.
> 
> **Commitment:** 12-15 hours/day, 6 days/week, for 12 months.

## What You'll Be Able to Do After 12 Months

| Skill | Description |
|-------|-------------|
| **Read GPU performance like a dashboard** | Memory-bound vs compute-bound, occupancy, stalls, launch overhead, data pipeline starvation |
| **Write kernels that are actually fast** | Reductions, scans, histograms, transpose, softmax, layernorm, attention—at near-hardware limits |
| **Use and extend the real ecosystem** | cuBLAS/cuDNN/NCCL, CUTLASS, Triton, torch.compile, CUDA graphs |
| **Build PyTorch CUDA extensions** | Forward + backward, tests + CI, proper benchmarking |
| **Optimize multi-GPU systems** | Overlap compute/comm, minimize syncs, scale to clusters |
| **Produce a public portfolio** | "Kernel Zoo" + 2-3 serious capstones that demonstrate expertise |

---

## Your Three Repositories

Throughout the bootcamp, you'll maintain three separate repositories:

### 1. `cuda-lab-notebook` (Daily Journal)
```
cuda-lab-notebook/
├── daily/
│   ├── 2025-01-15.md      # What you learned, what failed
│   ├── 2025-01-16.md
│   └── ...
├── microbench/
│   ├── bandwidth-tests/
│   ├── latency-tests/
│   └── roofline-data/
└── insights/
    ├── memory-coalescing.md
    ├── occupancy-myths.md
    └── ...
```

### 2. `kernel-zoo` (Your Kernel Library)
```
kernel-zoo/
├── primitives/
│   ├── reduction/
│   │   ├── naive.cu
│   │   ├── warp_shuffle.cu
│   │   ├── cub_reference.cu
│   │   ├── benchmark.py
│   │   └── README.md
│   ├── scan/
│   ├── histogram/
│   └── transpose/
├── gemm/
│   ├── naive/
│   ├── tiled/
│   ├── double_buffered/
│   ├── tensor_core/
│   └── cutlass_wrapper/
├── ml_ops/
│   ├── softmax/
│   ├── layernorm/
│   ├── attention/
│   └── fused_mlp/
└── benchmarks/
    ├── roofline.py
    ├── compare_implementations.py
    └── regression_tests.py
```

**Every kernel must have:**
- [ ] CPU reference implementation
- [ ] Correctness tests (random seeds, edge cases)
- [ ] Benchmark harness
- [ ] Performance targets and regression checks
- [ ] README explaining the optimization journey

### 3. `ml-ops-playground` (Integration with ML Stacks)
```
ml-ops-playground/
├── pytorch_extensions/
│   ├── fused_layernorm/
│   │   ├── csrc/
│   │   ├── setup.py
│   │   ├── test_gradients.py
│   │   └── benchmark.py
│   └── fused_attention/
├── triton_kernels/
│   ├── softmax.py
│   ├── layernorm.py
│   └── benchmark_vs_cuda.py
├── torch_compile_experiments/
│   ├── fusion_analysis/
│   └── custom_backends/
└── end_to_end/
    ├── transformer_block/
    └── inference_engine/
```

---

## 📚 Core Reference Documents (Required Daily Reading)

These are your **spine**—not optional reading:

| Document | Purpose | Link |
|----------|---------|------|
| **CUDA Programming Guide** | The *what* and *how* of CUDA | [nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) |
| **CUDA Best Practices Guide** | The *why* and *when* of performance | [nvidia.com](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) |
| **Daily Reference Spine** | Maps each phase to specific doc sections | [phase0/daily-reference-spine.md](phase0/daily-reference-spine.md) |
| **Library-First Guide** | When NOT to write custom kernels | [phase0/library-first-guide.md](phase0/library-first-guide.md) |

> **Key insight:** The difference between good and great CUDA developers is that great ones read the official docs weekly.

---

## Daily Cadence (Sustainable 15-Hour Structure)

| Hours | Activity | Purpose |
|-------|----------|---------|
| 2h | Theory/Reading | **Official docs + papers** (not optional!) |
| 5h | Implement Kernels | Correctness first, then optimize |
| 2h | Profiling + Optimization | Nsight Compute/Systems, roofline thinking |
| 2h | Expert Code Study | CUTLASS, FlashAttention, cuDNN patterns |
| 1h | Lab Notebook | Document what worked, what didn't |
| 2-3h | Active Recovery | Walk, gym, food, decompression (non-negotiable) |

**One day per week:** Lighter pace—review, cleanup, refactors, writeups, rest.

---

## The 12-Month Curriculum

### Phase 0: Foundation (Weeks 1-4) ✅ COMPLETE

**Goal:** Build, run, profile, and debug with confidence. Profiling becomes reflex.

**📁 [Phase 0 Materials →](phase0/README.md)** | **📝 [Checkpoint Quiz →](phase0/checkpoint-quiz.md)**

#### Week 1: Build System Mastery ✅
- [x] CMake + ninja for CUDA projects
- [x] Compiler flags: `-O3`, `-lineinfo`, `-arch=sm_XX`
- [x] Understanding PTX vs SASS
- [x] Setting up a reproducible benchmark harness

#### Week 2: Debugging Foundations ✅
- [x] compute-sanitizer (race detection, memory errors)
- [x] cuda-gdb basics
- [x] Understanding error codes and async error handling
- [x] **Nsight Systems timeline analysis** (profiling starts here!)

#### Week 3: Performance Analysis ✅
- [x] Nsight Compute kernel profiling
- [x] Memory metrics and bandwidth analysis
- [x] Compute metrics and occupancy
- [x] Roofline model understanding
- [x] Bottleneck identification
- [x] Systematic optimization workflow

#### Week 4: Project Templates ✅
- [x] Single-file quick-start template
- [x] Library template with clean C++ API
- [x] Application template with CLI
- [x] Benchmark framework with CSV/JSON output
- [x] Test framework with assertions
- [x] Complete all-in-one template

**Key Checkpoints:**
- [x] ✅ **Library-First**: Read [library-first-guide.md](phase0/library-first-guide.md)
- [x] ✅ **Daily Reference**: Read [daily-reference-spine.md](phase0/daily-reference-spine.md)
- [x] ✅ **Profiling Reflex**: You automatically `ncu` any kernel you write

**Deliverables:**
- [x] 6 production-ready project templates
- [x] Benchmark harness that reports GB/s, GFLOPS, and roofline position
- [x] Profiler-validated "hello CUDA" kernel

---

### Phase 1: CUDA Fundamentals (Weeks 5-8) ✅ COMPLETE

**Goal:** Write correct kernels and deeply understand the execution model.

**📁 [Phase 1 Materials →](phase1/README.md)** | **📝 [Checkpoint Quiz →](phase1/checkpoint-quiz.md)**

#### Week 5: Execution Model ✅
- [x] Threads, warps, blocks, grids—mental model
- [x] Indexing patterns (1D, 2D, 3D)
- [x] Grid-stride loops for arbitrary sizes
- [x] When kernels actually execute (streams, events)

#### Week 6: Memory Hierarchy ✅
- [x] Global memory: bandwidth, latency, coalescing
- [x] Shared memory: bank conflicts, padding
- [x] Registers: pressure, spilling to local memory
- [x] Constant and texture memory (when relevant)

#### Week 7: First Real Kernels ✅
- [x] Vector add (trivially parallel baseline)
- [x] SAXPY (memory-bound pattern)
- [x] Elementwise ops with different layouts
- [x] Reduction (6 versions: naive → warp shuffle)
- [x] Scan (Hillis-Steele, Blelloch, stream compaction)
- [x] Histogram (global → shared → warp aggregate)
- [x] Matrix multiply (naive → tiled → cuBLAS comparison)
- [x] **CUB comparison** (library-first validation)

#### Week 8: Synchronization & Hazards ✅
- [x] `__syncthreads()` correct usage
- [x] Warp-level primitives (shuffle, voting)
- [x] Atomic operations (patterns, contention reduction)
- [x] Memory fences and visibility
- [x] Cooperative groups API
- [x] Lock-free patterns

**Gate:** You can explain *why* a kernel is slow, not just that it is.

**Deliverables:**
- [x] Reduction kernel at >70% of memory bandwidth
- [x] Transpose kernel with bank conflict analysis (in Week 6)
- [x] CUB vs hand-written comparison

> **Note:** Week 7-8 expanded to cover warp primitives, scan, and histogram originally planned for Phase 3. This accelerates the learning path.

---

### Phase 2: Performance Mental Models (Weeks 9-12) ✅ COMPLETE

**Goal:** Measure and optimize with intent.

**📁 [Phase 2 Materials →](phase2/README.md)**

#### Week 9: Roofline Model ✅
- [x] Arithmetic intensity calculation
- [x] Memory bandwidth vs compute throughput
- [x] Plotting your kernels on the roofline
- [x] Identifying memory-bound vs compute-bound

#### Week 10: Occupancy Deep Dive ✅
- [x] Theoretical vs achieved occupancy
- [x] Why high occupancy ≠ high performance
- [x] Register usage and shared memory tradeoffs
- [x] Launch configuration experiments

#### Week 11: Profiling Mastery ✅
- [x] Nsight Compute: metrics that matter
- [x] Nsight Systems: timeline analysis
- [x] Identifying stalls, low utilization, memory issues
- [x] Profiling-driven optimization workflow

#### Week 12: Latency Hiding ✅
- [x] Instruction-level parallelism
- [x] Memory-level parallelism
- [x] Kernel launch overhead
- [x] When fusion matters (and when it doesn't)

**Gate:** Reduction and transpose hit meaningful fraction of peak bandwidth.

**Deliverables:**
- [x] Roofline plot of all Phase 1 kernels
- [x] Optimization report: before/after with profiler evidence
- [x] Blog post: "What I Learned About GPU Performance"

---

### Phase 3: Production Patterns (Weeks 13-16) ✅ COMPLETE

**Goal:** Master production-ready CUDA patterns: warp-level primitives, CUDA libraries, kernel fusion, and memory management.

**📁 [Phase 3 Materials →](phase3/README.md)**

**Library-First Checkpoint:** Before writing custom kernels, verify CUB/cuBLAS can't do it better. See [library-first-guide.md](phase0/library-first-guide.md).

#### Week 13: Warp-Level Programming ✅
- [x] Warp fundamentals (SIMT model, divergence)
- [x] Shuffle instructions (`__shfl_sync`, up, down, xor)
- [x] Warp reductions (sum, max, min without shared memory)
- [x] Warp scans (inclusive/exclusive prefix sums)
- [x] Vote functions (`__ballot_sync`, `__any_sync`, `__all_sync`)
- [x] Warp-level patterns (building blocks, dot product)

#### Week 14: CUDA Libraries ✅
- [x] cuBLAS basics (AXPY, DOT, GEMM)
- [x] cuBLAS advanced (batched ops, tensor cores)
- [x] CUB primitives (DeviceReduce, DeviceScan, DeviceSort)
- [x] CUB advanced (BlockReduce, WarpReduce, custom ops)
- [x] Thrust (device_vector, STL-like algorithms)
- [x] Library selection (decision framework, benchmarks)

#### Week 15: Kernel Fusion ✅
- [x] Fusion basics (why fuse, launch overhead, memory traffic)
- [x] Element-wise fusion (chained activations, GELU)
- [x] Reduction fusion (transform+reduce, softmax)
- [x] Tiled fusion (MatMul+bias+ReLU)
- [x] Producer-consumer patterns (register handoff, pipelines)
- [x] Fusion strategies (when to fuse, anti-patterns)

#### Week 16: Memory Management ✅
- [x] Memory pools (cudaMallocAsync, stream-ordered)
- [x] Pinned memory (cudaMallocHost, async transfers)
- [x] Zero-copy memory (mapped host memory, UVA)
- [x] Memory compaction (fragmentation, slab allocators)
- [x] Large data patterns (chunking, double/triple buffering)
- [x] Memory best practices (RAII, monitoring)

**Gate:** You can select between libraries and custom code, fuse kernels to reduce memory traffic.

**Deliverables:**
- [x] Warp shuffle-based reduction beating shared memory version
- [x] cuBLAS/CUB integration with proper error handling
- [x] Fused kernel showing 2x+ speedup over separate kernels

---

### Phase 4: Domain Applications (Weeks 17-20) ✅ COMPLETE

**Goal:** Apply CUDA skills to real-world domains: image processing, AI inference, physics simulation.

**📁 [Phase 4 Materials →](phase4/README.md)**

#### Week 17: Image Processing ✅
- [x] Image convolution kernels (naive → tiled)
- [x] Separable filters (2D → 1D + 1D)
- [x] Histogram computation and equalization
- [x] Image resizing (bilinear, bicubic interpolation)
- [x] Edge detection (Sobel, Canny)
- [x] NPP library comparison

#### Week 18: AI Inference Optimization ✅
- [x] Model loading and weight management
- [x] Batched inference patterns
- [x] Quantization (FP16, INT8 basics)
- [x] Layer fusion for inference
- [x] Memory planning and reuse
- [x] Tokens/sec benchmarking

#### Week 19: Physics Simulation ✅
- [x] N-body simulation (naive → optimized)
- [x] Particle systems
- [x] Spatial data structures (grids, BVH)
- [x] Collision detection
- [x] Fluid simulation basics
- [x] Real-time constraints

#### Week 20: Integration & Capstone ✅
- [x] Python bindings (pybind11, ctypes)
- [x] Multi-GPU data parallelism
- [x] Performance profiling workflow
- [x] End-to-end application
- [x] Documentation and packaging
- [x] Portfolio presentation

**Gate:** Complete end-to-end application demonstrating all Phase 1-4 skills.

**Deliverables:**
- [x] Image processing pipeline with 10x+ CPU speedup
- [x] Inference engine for simple model
- [x] Physics simulation at real-time rates
- [x] Capstone project with benchmarks and documentation

---

### Phase 5: GEMM Deep Dive (Weeks 21-28) ✅ COMPLETE

**Goal:** Understand the king of ML compute. Approach cuBLAS-level thinking.

**📁 [Phase 5 Materials →](phase5/README.md)**

**Library-First Checkpoint:** Always benchmark against cuBLAS first. Custom GEMM only for fusion.

#### Week 21-22: Naive to Tiled ✅
- [x] Naive matmul (baseline) + **cuBLAS comparison**
- [x] Tiled matmul with shared memory
- [x] Tile size selection and occupancy tradeoffs
- [x] Measuring achieved TFLOPS

#### Week 23-24: Advanced Tiling ✅
- [x] Double buffering (hide memory latency)
- [x] Register tiling (maximize compute per load)
- [x] Vectorized loads (float4)
- [x] Loop unrolling strategies

#### Week 25-26: Tensor Cores ✅
- [x] WMMA API (conceptual understanding)
- [x] MMA instructions (PTX level, optional)
- [x] Mixed precision (FP16 compute, FP32 accumulate)
- [x] When tensor cores win vs regular CUDA cores

#### Week 27-28: CUTLASS Introduction ✅
- [x] CUTLASS architecture: Tile, Epilogue, Mainloop
- [x] Using CUTLASS as a library
- [x] Understanding CUTLASS template parameters
- [x] Epilogue fusion (bias + activation)

**Gate:** Tiled GEMM is fast enough that remaining gap to cuBLAS is explainable.

**Deliverables:**
- [x] GEMM at >50% of cuBLAS performance
- [x] Performance analysis: where the remaining time goes
- [x] CUTLASS-based GEMM with fused epilogue

---

### Phase 6: Deep Learning Kernels (Weeks 29-32) ✅ COMPLETE

> **Note:** Condensed from 10 weeks to 4 weeks, focusing on core ML inference patterns.

**Goal:** Build the primitives that make transformers and modern nets fast.

**📁 [Phase 6 Materials →](phase6/README.md)**

#### Week 29: Quantization Fundamentals ✅
- [x] INT8/FP16 representation and scaling
- [x] Symmetric vs asymmetric quantization
- [x] Calibration methods (MinMax, Entropy, MSE)
- [x] Quantized GEMM with cuBLAS GemmEx

#### Week 30: Custom CUDA Operators ✅
- [x] PyTorch C++/CUDA extensions
- [x] Custom forward/backward kernels
- [x] Fused operations (LayerNorm+GELU)
- [x] TensorFlow custom ops

#### Week 31: Inference Optimization ✅
- [x] TensorRT engine building
- [x] Automatic layer fusion
- [x] Precision modes (FP32/FP16/INT8)
- [x] Custom TensorRT plugins

#### Week 32: Production Deployment ✅
- [x] Triton Inference Server basics
- [x] Model ensemble pipelines
- [x] Profiling and metrics (Prometheus)
- [x] Scaling strategies and best practices

**Gate:** You can deploy optimized models in production inference systems.

**Deliverables:**
- [x] Quantization pipeline for INT8 inference
- [x] Custom PyTorch CUDA extension
- [x] TensorRT-optimized model deployment

---

### Phase 7: DL Kernels & Attention (Weeks 33-40)

**Goal:** Master deep learning kernel optimization and attention mechanisms.

**📁 [Phase 7 Materials →](phase7/README.md)** *(in progress)*

**Library-First Checkpoint:** Check cuDNN first. Custom only for fusion or novel algorithms.

#### Week 33-34: Softmax & LayerNorm
- [ ] Numerical stability (max subtraction)
- [ ] Online softmax (single pass)
- [ ] Welford's online algorithm for variance
- [ ] LayerNorm / RMSNorm forward and backward
- [ ] **Benchmark vs cuDNN**

#### Week 35-36: Attention Building Blocks
- [ ] QK^T computation (batched GEMM)
- [ ] Masking (causal, padding)
- [ ] Softmax over attention scores
- [ ] PV computation and output projection

#### Week 37-38: FlashAttention Study
- [ ] IO-aware algorithm design
- [ ] Tiling strategy for attention
- [ ] Online softmax in attention context
- [ ] Reading and understanding the paper + code

#### Week 39-40: Kernel Fusion Strategies
- [ ] Fused bias + dropout + residual
- [ ] Fused attention patterns
- [ ] Memory traffic reduction analysis
- [ ] When fusion helps vs hurts

**Gate:** You can explain why FlashAttention works and implement a simplified version.

**Deliverables:**
- [ ] Fused softmax at near-bandwidth limit
- [ ] LayerNorm forward + backward with gradient checks
- [ ] Attention mini-implementation with performance analysis

---

### Phase 8: ML Stack & Multi-GPU (Weeks 41-48)

**Goal:** Ship kernels as usable components and scale to multi-GPU systems.

**📁 [Phase 8 Materials →](phase8/README.md)** *(reserved for future)*

#### Week 41-42: PyTorch C++ Extensions
- [ ] Extension setup (setup.py, CMake)
- [ ] Tensor accessors and data types
- [ ] Autograd integration (forward/backward)
- [ ] Gradient checking and validation

#### Week 43-44: Triton & torch.compile
- [ ] Triton programming model
- [ ] Converting CUDA kernels to Triton
- [ ] How Inductor generates kernels
- [ ] **When Triton beats hand-written CUDA**

#### Week 45-46: NCCL Fundamentals
- [ ] All-reduce, reduce-scatter, all-gather
- [ ] Ring vs tree algorithms
- [ ] Bandwidth and latency characteristics
- [ ] NCCL debugging

#### Week 47-48: Multi-GPU Strategies
- [ ] Compute/communication overlap
- [ ] Streams and events for coordination
- [ ] Gradient bucketing and fusion
- [ ] Pipeline parallelism concepts

**Gate:** Drop your custom op into a model and measure end-to-end speedup at scale.

**Deliverables:**
- [ ] PyTorch extension: fused LayerNorm (forward + backward)
- [ ] Same op in Triton + performance comparison
- [ ] Multi-GPU scaling benchmark

---

### Phase 9: Capstones & Portfolio (Weeks 49-52)

**Goal:** Complete portfolio-quality projects demonstrating mastery.

**📁 [Capstone Materials →](capstones/README.md)** *(reserved for future)*

#### Week 49-50: Capstone Projects
- [ ] CUDA graph capture and replay
- [ ] Complete 2-3 portfolio projects
- [ ] End-to-end optimized inference engine
- [ ] Custom attention implementation

#### Week 51-52: Documentation & Polish
- [ ] Documentation and writeups
- [ ] Performance regression suite
- [ ] Blog posts and portfolio presentation
- [ ] Career preparation

**Deliverables:**
- [ ] 3 portfolio-quality capstone projects
- [ ] Public GitHub with documented work
- [ ] Technical blog posts

---

## Curriculum Summary (52 Weeks)

| Phase | Weeks | Focus | Status |
|-------|-------|-------|--------|
| 0 | 1-4 | Foundation: Build, Debug, Profile, Templates | ✅ Complete |
| 1 | 5-8 | CUDA Fundamentals: Execution & Memory Model | ✅ Complete |
| 2 | 9-12 | Performance: Roofline, Occupancy, Profiling | ✅ Complete |
| 3 | 13-16 | Production: Warps, Libraries, Fusion, Memory | ✅ Complete |
| 4 | 17-20 | Applications: Image, AI Inference, Physics | ✅ Complete |
| 5 | 21-28 | GEMM: Tiling, Tensor Cores, CUTLASS | ✅ Complete |
| 6 | 29-32 | ML Inference: Quantization, TensorRT, Triton Server | ✅ Complete |
| 7 | 33-40 | DL Kernels: Softmax, LayerNorm, Attention, FlashAttention | 🔄 In Progress |
| 8 | 41-48 | ML Stack: PyTorch, Triton, NCCL, Multi-GPU | ⏳ Reserved |
| 9 | 49-52 | Capstones: Portfolio Projects | ⏳ Reserved |

---

## Hardware Considerations

### Target GPUs by Phase

| Phase | Minimum GPU | Recommended | Notes |
|-------|-------------|-------------|-------|
| 0-2 | Any CUDA GPU | T4 / RTX 3060 | Fundamentals work on anything |
| 3-4 | T4 / V100 | A100 | GEMM needs tensor cores |
| 5-6 | A100 | A100 / H100 | Large attention operations |
| 7-8 | Multi-GPU setup | 2-8× A100 | NCCL, scaling experiments |

### Hardware-Specific Topics

#### Ampere (A100)
- [ ] TF32 (faster FP32-like compute)
- [ ] BF16 native support
- [ ] Sparse tensor cores (2:4 sparsity)
- [ ] 40GB vs 80GB memory configurations

#### Hopper (H100)
- [ ] FP8 (E4M3, E5M2) for inference
- [ ] Transformer Engine
- [ ] Thread block clusters
- [ ] Distributed shared memory

#### Ada Lovelace (RTX 4090)
- [ ] Consumer vs datacenter differences
- [ ] FP8 support
- [ ] Power efficiency considerations

---

## Recommended Reading Schedule

### Books
| When | Book | Focus |
|------|------|-------|
| Phase 0-1 | "Programming Massively Parallel Processors" (Kirk & Hwu) | Foundations |
| Phase 2-3 | "CUDA Handbook" (Wilt) | Deep reference |
| Phase 4 | CUTLASS documentation | GEMM patterns |

### Papers
| When | Paper | Why |
|------|-------|-----|
| Phase 5 | "FlashAttention" (Dao et al.) | IO-aware algorithm design |
| Phase 5 | "Online Softmax" | Single-pass numerical stability |
| Phase 7 | "Megatron-LM" | Multi-GPU training patterns |

### Codebases to Study
| When | Codebase | What to Learn |
|------|----------|---------------|
| Phase 3-4 | CUB library | Optimized primitives |
| Phase 4 | CUTLASS | GEMM architecture |
| Phase 5 | FlashAttention repo | Production attention |
| Phase 6 | Triton tutorials | High-level kernel writing |
| Phase 7 | DeepSpeed | Distributed training |

---

## Advanced Topics (For Extended Study)

These topics extend beyond the core 12-month curriculum:

### Quantization & Low Precision
- [ ] INT8 inference (calibration, accuracy)
- [ ] FP8 training and inference
- [ ] INT4/INT3 for extreme compression
- [ ] Quantization-aware training

### Sparsity
- [ ] Structured sparsity (2:4 for tensor cores)
- [ ] Unstructured sparsity
- [ ] Sparse attention patterns
- [ ] Pruning strategies

### Custom Memory Management
- [ ] CUDA memory pools
- [ ] Custom allocators
- [ ] Memory-mapped I/O
- [ ] Unified memory for large models

### Emerging Hardware
- [ ] AMD GPUs (ROCm, HIP)
- [ ] Intel GPUs (oneAPI, SYCL)
- [ ] TPUs and custom accelerators
- [ ] Comparison and portability

---

## Success Metrics

### By Month 3
- [ ] Can write correct kernels for basic operations
- [ ] Understand memory hierarchy and coalescing
- [ ] Can use Nsight to identify performance issues

### By Month 6
- [ ] Scan and histogram at high efficiency
- [ ] GEMM within 2× of cuBLAS
- [ ] Can explain roofline position of any kernel

### By Month 9
- [ ] Attention kernels working and optimized
- [ ] PyTorch extension with autograd
- [ ] Can compare CUDA vs Triton trade-offs

### By Month 12
- [ ] Multi-GPU scaling understood
- [ ] 3 portfolio-quality capstones complete
- [ ] Ready for GPU performance engineering roles

---

## Community & Resources

### Where to Ask Questions
- NVIDIA Developer Forums
- CUDA Programming Subreddit
- GPU Mode Discord
- Twitter/X GPU performance community

### Conferences to Follow
- GTC (NVIDIA's conference)
- MLSys
- ISCA, MICRO (architecture)

### Companies Hiring for This Skillset
- NVIDIA, AMD, Intel (hardware vendors)
- Meta, Google, Microsoft, Amazon (ML infra)
- Anthropic, OpenAI, Cohere (AI labs)
- Trading firms (low-latency compute)

---

## Relationship to 18-Week Tutorial

| Aspect | 18-Week Tutorial | This Bootcamp |
|--------|------------------|---------------|
| **Purpose** | Learn CUDA concepts | Master CUDA for career |
| **Audience** | Anyone curious | ML researchers going deep |
| **Commitment** | Part-time | Full-time |
| **Output** | Understanding | Portfolio |
| **Overlap** | Weeks 1-12 cover similar ground | Goes much deeper |

**Recommended Path:**
1. Complete [18-Week Tutorial](../learning-path/README.md) for conceptual foundation
2. Decide if GPU performance engineering is your path
3. If yes, use this bootcamp as your roadmap

---

## Final Note

This bootcamp is brutal by design. It's meant for people who want to become *experts*, not just practitioners. The investment is significant:

- 12-15 hours/day × 6 days/week × 52 weeks = **4,000-5,000 hours**

But the outcome is clear: you'll be able to compete for the most demanding GPU programming roles in the industry.

**The 18-week tutorial teaches you to fish. This bootcamp makes you a marine biologist.**
