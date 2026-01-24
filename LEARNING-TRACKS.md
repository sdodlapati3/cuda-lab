# cuda-lab Learning Tracks

> **Choose your path based on your goals and time commitment**

---

## 🗺️ Track Overview

```
                            ┌─────────────────────────────────────────┐
                            │           cuda-lab LEARNING             │
                            └─────────────────────────────────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────┐
         │                                    │                                │
         ▼                                    ▼                                ▼
┌─────────────────┐               ┌─────────────────┐              ┌─────────────────┐
│  FOUNDATION     │               │   MASTERY       │              │  NESAP/HPC      │
│   Track 1       │───────────────│   Track 2       │──────────────│   Track 3       │
│  learning-path/ │               │   bootcamp/     │              │   NEW           │
└─────────────────┘               └─────────────────┘              └─────────────────┘
         │                                    │                                │
         ▼                                    ▼                                ▼
• 18 weeks part-time       • 52 weeks full-time        • 12 weeks intensive
• 4-6 hrs/day              • 12-15 hrs/day             • Focused on HPC/ML
• Colab-compatible         • HPC cluster required      • Profiling mastery
• CUDA fundamentals        • Expert-level mastery      • Scientific ML
```

---

## 📘 Track 1: Foundation (learning-path/)

**For:** Anyone wanting to learn CUDA programming from scratch.

| Aspect | Details |
|--------|---------|
| **Duration** | 18 weeks (part-time) |
| **Time** | 4-6 hours/day, 5 days/week |
| **Total Hours** | ~400 hours |
| **Hardware** | Google Colab T4 (free) or local GPU |
| **Outcome** | Working CUDA proficiency |

### What You'll Learn
- GPU architecture and execution model
- Memory hierarchy and optimization
- Parallel algorithms (reduction, scan, histogram)
- Matrix operations and cuBLAS
- Streams, events, CUDA graphs
- Multi-GPU basics

### Path Structure
```
Weeks 1-6:   Foundations (architecture, memory, parallel patterns)
Weeks 7-12:  Optimization (profiling, streams, graphs, multi-GPU)
Weeks 13-16: Advanced (unified memory, virtual memory, sync)
Weeks 17-18: HPC Features (IPC, textures, MIG) - requires cluster
```

### Start Here
📁 [learning-path/README.md](learning-path/README.md)

### Prerequisites
- C/C++ programming basics
- Basic understanding of computer architecture
- No prior GPU experience required

---

## 📕 Track 2: Mastery (bootcamp/)

**For:** ML engineers committed to becoming GPU performance experts.

| Aspect | Details |
|--------|---------|
| **Duration** | 52 weeks (full-time) |
| **Time** | 12-15 hours/day, 6 days/week |
| **Total Hours** | ~4,000-5,000 hours |
| **Hardware** | HPC cluster with A100/H100 required |
| **Outcome** | Expert-level GPU performance engineering |

### What You'll Learn
- Everything in Track 1, but much deeper
- GEMM from scratch → CUTLASS integration
- FlashAttention and ML kernel optimization
- PyTorch C++/CUDA extensions
- Triton programming
- Multi-GPU with NCCL
- Production deployment patterns

### Path Structure
```
Phase 0 (Weeks 1-4):    Foundation - Build, debug, profile
Phase 1 (Weeks 5-8):    CUDA Fundamentals - Execution model
Phase 2 (Weeks 9-12):   Performance - Roofline, occupancy
Phase 3 (Weeks 13-16):  Production - Warps, libraries, fusion
Phase 4 (Weeks 17-20):  Applications - Image, AI, physics
Phase 5 (Weeks 21-28):  GEMM - Tiling, tensor cores, CUTLASS
Phase 6 (Weeks 29-32):  ML Inference - Quantization, TensorRT
Phase 7 (Weeks 33-40):  DL Kernels - Attention, FlashAttention
Phase 8 (Weeks 41-48):  ML Stack - PyTorch, Triton, NCCL
Phase 9 (Weeks 49-52):  Capstones - Portfolio projects
```

### Start Here
📁 [bootcamp/README.md](bootcamp/README.md)

### Prerequisites
- **Complete Track 1** or equivalent experience
- Full-time commitment (this is a career investment)
- Access to multi-GPU HPC system

---

## 📗 Track 3: NESAP/HPC Preparation (NEW)

**For:** Targeting HPC centers, national labs, scientific ML roles.

| Aspect | Details |
|--------|---------|
| **Duration** | 12 weeks (intensive) |
| **Time** | 6-8 hours/day, 5 days/week |
| **Total Hours** | ~400 hours |
| **Hardware** | Multi-GPU cluster required |
| **Outcome** | NESAP/HPC interview ready |

### What You'll Learn
- Performance profiling mastery (Nsight Systems/Compute)
- HPC workflows (Slurm, checkpointing, containers)
- Scientific ML (PINNs, surrogate models, UQ)
- Distributed training optimization
- Scaling benchmarks and efficiency metrics
- Energy-aware computing

### Focus Areas (NESAP Skill Alignment)

| Week | Focus | NESAP Skill |
|------|-------|-------------|
| 1-2 | Profiling Mastery | Performance Profiling |
| 3-4 | Distributed Training | DDP, FSDP, NCCL |
| 5-6 | HPC Workflows | Slurm, containers, checkpointing |
| 7-8 | Scientific ML | PINNs, surrogate models |
| 9-10 | Data Pipelines | Large-scale I/O |
| 11-12 | Scaling & Benchmarks | Efficiency metrics |

### Path Structure
```
profiling-lab/     → Weeks 1-2
bootcamp/phase8/   → Weeks 3-4 (distributed training)
hpc-lab/           → Weeks 5-6
scientific-ml/     → Weeks 7-8
data-pipelines/    → Weeks 9-10
benchmarks/        → Weeks 11-12
```

### Start Here
📁 [profiling-lab/README.md](profiling-lab/README.md) (coming soon)
📁 [hpc-lab/README.md](hpc-lab/README.md) (coming soon)

### Prerequisites
- Track 1 completion (or equivalent)
- Access to HPC cluster (NERSC, university cluster, etc.)
- Python + PyTorch proficiency

---

## 🔄 Track Relationships

### Can I skip Track 1?

**If you can answer YES to all:**
- [ ] I can write a CUDA kernel from scratch without reference
- [ ] I understand coalesced memory access and bank conflicts
- [ ] I can implement parallel reduction with warp shuffles
- [ ] I've used shared memory for tiled algorithms
- [ ] I know the difference between streams and events

**Then:** Start at Track 2 or Track 3

### Can I do Track 2 and Track 3 together?

**Not recommended.** Track 2 is a full-time commitment. However:
- If targeting NESAP: Do Track 1 → Track 3
- If building a career in GPU performance: Do Track 1 → Track 2
- Track 3 content can supplement Track 2 during relevant phases

---

## 📊 Comparison Table

| Aspect | Track 1 | Track 2 | Track 3 |
|--------|---------|---------|---------|
| Duration | 18 weeks | 52 weeks | 12 weeks |
| Intensity | Part-time | Full-time | Intensive |
| Hardware | Colab OK | HPC required | HPC required |
| CUDA depth | Working | Expert | Working+ |
| ML integration | Light | Heavy | Scientific |
| Profiling | Basics | Advanced | Mastery |
| HPC workflows | Minimal | Some | Extensive |
| Portfolio output | Basic | Comprehensive | Targeted |
| Career target | General | Performance engineering | HPC/national labs |

---

## 🎯 Quick Decision Guide

**I want to...**

| Goal | Recommended Track |
|------|-------------------|
| Learn CUDA basics | Track 1 |
| Become a CUDA expert | Track 1 → Track 2 |
| Work at NVIDIA/AMD | Track 1 → Track 2 |
| Work at AI labs (Anthropic, OpenAI) | Track 1 → Track 2 |
| Work at HPC centers (NERSC, ORNL) | Track 1 → Track 3 |
| Do scientific ML research | Track 1 → Track 3 |
| Optimize ML training at scale | Track 1 → Track 2 (Phase 7-8) |
| Build a quick portfolio | Track 1 (Weeks 1-12) |

---

## 📚 Supporting Materials (All Tracks)

| Resource | Description |
|----------|-------------|
| [cuda-programming-guide/](cuda-programming-guide/) | Reference documentation (CUDA 13.1) |
| [practice/](practice/) | Hands-on exercises by topic |
| [notes/cuda-quick-reference.md](notes/cuda-quick-reference.md) | Syntax cheatsheet |
| [docs/modern-gpu-ecosystem.md](docs/modern-gpu-ecosystem.md) | When to use Triton/cuBLAS/etc. |

---

*Choose your track and start learning! 🚀*
