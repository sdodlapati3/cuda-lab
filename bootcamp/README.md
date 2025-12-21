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

### Phase 0: Foundation (Weeks 1-4)

**Goal:** Build, run, profile, and debug with confidence. Profiling becomes reflex.

**📁 [Phase 0 Materials →](phase0/README.md)**

#### Week 1: Build System Mastery
- [ ] CMake + ninja for CUDA projects
- [ ] Compiler flags: `-O3`, `-lineinfo`, `-arch=sm_XX`
- [ ] Understanding PTX vs SASS
- [ ] Setting up a reproducible benchmark harness

#### Week 2: Debugging Foundations
- [ ] compute-sanitizer (race detection, memory errors)
- [ ] cuda-gdb basics
- [ ] Understanding error codes and async error handling
- [ ] **Nsight Systems timeline analysis** (profiling starts here!)

#### Week 3: Performance Analysis
- [ ] Nsight Compute kernel profiling
- [ ] Memory metrics and bandwidth analysis
- [ ] Compute metrics and occupancy
- [ ] Roofline model understanding
- [ ] Bottleneck identification
- [ ] Systematic optimization workflow

#### Week 4: Project Templates
- [ ] Single-file quick-start template
- [ ] Library template with clean C++ API
- [ ] Application template with CLI
- [ ] Benchmark framework with CSV/JSON output
- [ ] Test framework with assertions
- [ ] Complete all-in-one template

**Key Checkpoints:**
- [ ] ✅ **Library-First**: Read [library-first-guide.md](phase0/library-first-guide.md)
- [ ] ✅ **Daily Reference**: Read [daily-reference-spine.md](phase0/daily-reference-spine.md)
- [ ] ✅ **Profiling Reflex**: You automatically `ncu` any kernel you write

**Deliverables:**
- [ ] 6 production-ready project templates
- [ ] Benchmark harness that reports GB/s, GFLOPS, and roofline position
- [ ] Profiler-validated "hello CUDA" kernel

---

### Phase 1: CUDA Fundamentals (Weeks 5-8)

**Goal:** Write correct kernels and deeply understand the execution model.

#### Week 5: Execution Model
- [ ] Threads, warps, blocks, grids—mental model
- [ ] Indexing patterns (1D, 2D, 3D)
- [ ] Grid-stride loops for arbitrary sizes
- [ ] When kernels actually execute (streams, events)

#### Week 6: Memory Hierarchy
- [ ] Global memory: bandwidth, latency, coalescing
- [ ] Shared memory: bank conflicts, padding
- [ ] Registers: pressure, spilling to local memory
- [ ] Constant and texture memory (when relevant)

#### Week 7: First Real Kernels
- [ ] Vector add (trivially parallel baseline)
- [ ] SAXPY (memory-bound pattern)
- [ ] Elementwise ops with different layouts
- [ ] Simple reduction (sum, max)

#### Week 8: Synchronization & Hazards
- [ ] `__syncthreads()` correct usage
- [ ] Race conditions and data hazards
- [ ] Atomic operations (when necessary, when to avoid)
- [ ] Matrix transpose (slow first, then improve)

**Gate:** You can explain *why* a kernel is slow, not just that it is.

**Deliverables:**
- [ ] Reduction kernel at >70% of memory bandwidth
- [ ] Transpose kernel with bank conflict analysis
- [ ] Written explanation of coalescing patterns

---

### Phase 2: Performance Mental Models (Weeks 9-12)

**Goal:** Measure and optimize with intent.

#### Week 9: Roofline Model
- [ ] Arithmetic intensity calculation
- [ ] Memory bandwidth vs compute throughput
- [ ] Plotting your kernels on the roofline
- [ ] Identifying memory-bound vs compute-bound

#### Week 10: Occupancy Deep Dive
- [ ] Theoretical vs achieved occupancy
- [ ] Why high occupancy ≠ high performance
- [ ] Register usage and shared memory tradeoffs
- [ ] Launch configuration experiments

#### Week 11: Profiling Mastery
- [ ] Nsight Compute: metrics that matter
- [ ] Nsight Systems: timeline analysis
- [ ] Identifying stalls, low utilization, memory issues
- [ ] Profiling-driven optimization workflow

#### Week 12: Latency Hiding
- [ ] Instruction-level parallelism
- [ ] Memory-level parallelism
- [ ] Kernel launch overhead
- [ ] When fusion matters (and when it doesn't)

**Gate:** Reduction and transpose hit meaningful fraction of peak bandwidth.

**Deliverables:**
- [ ] Roofline plot of all Phase 1 kernels
- [ ] Optimization report: before/after with profiler evidence
- [ ] Blog post: "What I Learned About GPU Performance"

---

### Phase 3: Core Parallel Primitives (Weeks 13-18)

**Goal:** Implement patterns that appear everywhere in GPU computing.

**Library-First Checkpoint:** Before implementing, verify CUB can't do it better. See [library-first-guide.md](phase0/library-first-guide.md).

#### Week 13-14: Warp-Level Programming
- [ ] Warp shuffle instructions (`__shfl_sync`, `__shfl_down_sync`)
- [ ] Warp-level reductions (no shared memory)
- [ ] Warp vote functions (`__ballot_sync`, `__any_sync`)
- [ ] Combining warp and block-level operations

#### Week 15-16: Scan (Prefix Sum)
- [ ] Inclusive vs exclusive scan
- [ ] Hillis-Steele vs Blelloch algorithms
- [ ] Block-level scan
- [ ] Multi-block scan with decoupled look-back
- [ ] **Benchmark against CUB::DeviceScan**

#### Week 17-18: Histograms & Advanced Atomics
- [ ] Naive atomic histogram (and why it's slow)
- [ ] Privatization strategies
- [ ] Warp-aggregated atomics
- [ ] Stream compaction (using scan)
- [ ] **Benchmark against CUB::DeviceHistogram**

**Gate:** Scan and histogram with correct performance reasoning.

**Deliverables:**
- [ ] Scan at >80% of theoretical bandwidth
- [ ] Histogram comparison: naive vs optimized vs CUB (speedup analysis)
- [ ] Written analysis of atomic contention

---

### Phase 4: GEMM Deep Dive (Weeks 19-26)

**Goal:** Understand the king of ML compute. Approach cuBLAS-level thinking.

**Library-First Checkpoint:** Always benchmark against cuBLAS first. Custom GEMM only for fusion.

#### Week 19-20: Naive to Tiled
- [ ] Naive matmul (baseline) + **cuBLAS comparison**
- [ ] Tiled matmul with shared memory
- [ ] Tile size selection and occupancy tradeoffs
- [ ] Measuring achieved TFLOPS

#### Week 21-22: Advanced Tiling
- [ ] Double buffering (hide memory latency)
- [ ] Register tiling (maximize compute per load)
- [ ] Vectorized loads (float4)
- [ ] Loop unrolling strategies

#### Week 23-24: Tensor Cores
- [ ] WMMA API (conceptual understanding)
- [ ] MMA instructions (PTX level, optional)
- [ ] Mixed precision (FP16 compute, FP32 accumulate)
- [ ] When tensor cores win vs regular CUDA cores

#### Week 25-26: CUTLASS Introduction
- [ ] CUTLASS architecture: Tile, Epilogue, Mainloop
- [ ] Using CUTLASS as a library
- [ ] Understanding CUTLASS template parameters
- [ ] Epilogue fusion (bias + activation)

**Gate:** Tiled GEMM is fast enough that remaining gap to cuBLAS is explainable.

**Deliverables:**
- [ ] GEMM at >50% of cuBLAS performance
- [ ] Performance analysis: where the remaining time goes
- [ ] CUTLASS-based GEMM with fused epilogue

---

### Phase 5: Deep Learning Kernels (Weeks 27-36)

**Goal:** Build the primitives that make transformers and modern nets fast.

**Library-First Checkpoint:** Check cuDNN first. Custom only for fusion or novel algorithms.

#### Week 27-28: Softmax
- [ ] Numerical stability (max subtraction)
- [ ] Online softmax (single pass)
- [ ] Memory traffic analysis
- [ ] Fused softmax (no intermediate storage)
- [ ] **Benchmark vs cuDNN softmax**

#### Week 29-30: LayerNorm / RMSNorm
- [ ] Mean and variance computation
- [ ] Welford's online algorithm
- [ ] Forward pass optimization
- [ ] Backward pass (gradient computation)
- [ ] **Benchmark vs cuDNN layer normalization**

#### Week 31-32: Attention Building Blocks
- [ ] QK^T computation (batched GEMM)
- [ ] Masking (causal, padding)
- [ ] Softmax over attention scores
- [ ] PV computation

#### Week 33-34: Kernel Fusion Strategies
- [ ] Fused bias + dropout + residual
- [ ] Fused attention patterns
- [ ] Memory traffic reduction analysis
- [ ] When fusion helps vs hurts

#### Week 35-36: FlashAttention Study
- [ ] IO-aware algorithm design
- [ ] Tiling strategy for attention
- [ ] Online softmax in attention context
- [ ] Reading and understanding the paper + code

**Gate:** You can explain why FlashAttention works and implement a simplified version.

**Deliverables:**
- [ ] Fused softmax at near-bandwidth limit
- [ ] LayerNorm forward + backward with gradient checks
- [ ] Attention mini-implementation with performance analysis

---

### Phase 6: ML Stack Integration (Weeks 37-44)

**Goal:** Ship kernels as usable components in real frameworks.

#### Week 37-38: PyTorch C++ Extensions
- [ ] Extension setup (setup.py, CMake)
- [ ] Tensor accessors and data types
- [ ] Error handling and checks
- [ ] Building and installing

#### Week 39-40: Autograd Integration
- [ ] Forward function implementation
- [ ] Backward function implementation
- [ ] Gradient checking and validation
- [ ] Handling edge cases (zero-size tensors, etc.)

#### Week 41-42: Triton
- [ ] Triton programming model
- [ ] Converting CUDA kernels to Triton
- [ ] Auto-tuning with Triton
- [ ] **When Triton beats hand-written CUDA** (and vice versa)

#### Week 43-44: torch.compile / Inductor
- [ ] How Inductor generates kernels
- [ ] Fusion analysis
- [ ] Custom backends
- [ ] When compilation fails and why

**Gate:** Drop your custom op into a model and measure end-to-end speedup.

**Deliverables:**
- [ ] PyTorch extension: fused LayerNorm (forward + backward)
- [ ] Same op in Triton + performance comparison
- [ ] Benchmark suite with regression tests

---

### Phase 7: Multi-GPU & Systems (Weeks 45-50)

**Goal:** Remove bottlenecks that appear at scale.

#### Week 45-46: NCCL Fundamentals
- [ ] All-reduce, reduce-scatter, all-gather
- [ ] Ring vs tree algorithms
- [ ] Bandwidth and latency characteristics
- [ ] NCCL debugging

#### Week 47-48: Overlap Strategies
- [ ] Compute/communication overlap
- [ ] Streams and events for coordination
- [ ] Gradient bucketing and fusion
- [ ] Pipeline parallelism concepts

#### Week 49-50: CUDA Graphs & Memory Management
- [ ] CUDA graph capture and replay
- [ ] When graphs help (inference, repetitive training)
- [ ] Memory allocator behavior
- [ ] Fragmentation and pool strategies

**Gate:** Explain why scaling plateaus and how to attack it.

**Deliverables:**
- [ ] Multi-GPU benchmark: all-reduce scaling curve
- [ ] Overlap demonstration: compute + comm speedup
- [ ] CUDA graph capture for training step

---

### Phase 8: Capstones (Weeks 51-52)

**Choose 2 major + 1 "sharp tool" capstone:**

#### Capstone A: Transformer Kernel Pack
- [ ] Fused softmax (optimized)
- [ ] LayerNorm (forward + backward)
- [ ] Fused MLP (GEMM + bias + activation)
- [ ] PyTorch extension with benchmarks
- [ ] Documentation and usage examples

#### Capstone B: Inference Microengine
- [ ] CUDA graphs for steady-state inference
- [ ] Memory planning and reuse
- [ ] Tokens/sec benchmarking
- [ ] Comparison with baseline PyTorch

#### Capstone C: Benchmark & Evaluation Framework
- [ ] CUDA vs Triton vs PyTorch comparison
- [ ] Latency vs throughput analysis
- [ ] Multi-GPU support (T4/A100/H100)
- [ ] Automated regression detection

**Final Output:**
- [ ] Public repository with clean code
- [ ] Write-up per capstone (performance plots, lessons learned)
- [ ] Blog post or video explaining your work

---

## Curriculum Summary (52 Weeks)

| Phase | Weeks | Focus |
|-------|-------|-------|
| 0 | 1-4 | Foundation: Build, Debug, Profile, Templates |
| 1 | 5-8 | CUDA Fundamentals: Execution & Memory Model |
| 2 | 9-12 | Performance: Roofline, Occupancy, Profiling |
| 3 | 13-18 | Primitives: Warps, Scan, Histogram |
| 4 | 19-26 | GEMM: Tiling, Tensor Cores, CUTLASS |
| 5 | 27-36 | DL Kernels: Softmax, LayerNorm, Attention |
| 6 | 37-44 | ML Stack: PyTorch, Triton, torch.compile |
| 7 | 45-50 | Systems: NCCL, Overlap, CUDA Graphs |
| 8 | 51-52 | Capstones: Portfolio Projects |

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
