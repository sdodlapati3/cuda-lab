# Daily Reference Spine

> **The two documents you should have open every day of this bootcamp.**

## The Official CUDA Documentation

These are not optional reading—they are your primary reference:

### 1. CUDA Programming Guide
**URL:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/

This is the **what** and **how** of CUDA:
- Execution model (threads, warps, blocks, grids)
- Memory hierarchy
- Language extensions
- Hardware capabilities

### 2. CUDA Best Practices Guide
**URL:** https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

This is the **why** and **when** of performance:
- APOD cycle (Assess, Parallelize, Optimize, Deploy)
- Memory optimization
- Execution configuration
- Instruction optimization

---

## How to Use This Document

Each phase below maps to specific sections of the official docs. 

**Daily practice:**
1. Before coding, read the relevant section
2. While coding, have the docs open in a tab
3. After coding, re-read to fill gaps you discovered

---

## Phase 0: Foundation → Official Doc Mapping

### Week 1: Build System
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Compilation | Ch. 3: Compilation with NVCC | — |
| PTX vs SASS | Ch. 3.1: Compilation Workflow | — |
| Compute capability | Appendix H: Compute Capabilities | Ch. 2.1: Assess |

### Week 2: Debugging
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Error handling | Ch. B.17: Error Checking | Ch. 8: Error Handling |
| Async errors | Ch. 3.2.9: Error Handling | — |

### Week 3: Performance Analysis
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Profiling | — | Ch. 2.2: APOD Cycle |
| Performance limiters | — | Ch. 3: Performance Metrics |

### Week 4: Project Templates
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Project structure | Ch. 3: Compilation | Ch. 2: Heterogeneous Computing |

---

## Phase 1: CUDA Fundamentals → Official Doc Mapping

### Week 3: Execution Model
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Thread hierarchy | Ch. 2.2: Thread Hierarchy | — |
| Kernel functions | Ch. 2.1: Kernels | — |
| Grid/block dims | Ch. 2.2: Thread Hierarchy | Ch. 10.1: Execution Configuration |

### Week 4: Memory Hierarchy
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Memory types | Ch. 5: Memory Hierarchy | Ch. 5: Memory Optimizations |
| Global memory | Ch. 5.3: Device Memory | Ch. 5.2: Device Memory Access |
| Shared memory | Ch. 5.4: Shared Memory | Ch. 5.3: Shared Memory |
| Coalescing | — | Ch. 5.2.1: Coalesced Access |
| Bank conflicts | — | Ch. 5.3.1: Shared Memory Banks |

### Week 5: First Kernels
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Kernel launch | Ch. 2.1: Kernels | Ch. 10.1: Launch Config |
| Built-in variables | Ch. 4.1: Built-in Variables | — |

### Week 6: Synchronization
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| __syncthreads() | Ch. 2.3: Synchronization | Ch. 7.1: Synchronization |
| Atomics | Ch. 7.14: Atomic Functions | Ch. 7.2: Atomic Functions |

---

## Phase 2: Performance → Official Doc Mapping

### Week 7: Roofline
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Bandwidth | Ch. 5.3: Device Memory | Ch. 3.1: Bandwidth |
| Throughput | — | Ch. 3.2: Throughput |

### Week 8: Occupancy
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Occupancy | Ch. 5.2.3: Maximize Utilization | Ch. 10.2: Occupancy |
| Register usage | Ch. 5.3.2: Register Memory | Ch. 10.3: Hiding Latency |

### Week 9: Profiling
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Nsight tools | — | Ch. 2.2: Profile |
| Metrics | — | Ch. 3: Performance Metrics |

### Week 10: Latency Hiding
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| ILP | — | Ch. 10.3: Hiding Latency |
| MLP | — | Ch. 10.3: Hiding Latency |

---

## Phase 3: Parallel Primitives → Official Doc Mapping

### Weeks 11-12: Warp-Level
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Warp shuffle | Ch. 7.22: Warp Shuffle | — |
| Warp vote | Ch. 7.21: Warp Vote | — |
| Cooperative groups | Ch. 9: Cooperative Groups | — |

### Weeks 13-14: Scan
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Parallel prefix | — | — (use CUB docs) |

### Weeks 15-16: Histogram
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Atomics | Ch. 7.14: Atomic Functions | Ch. 7.2: Atomic Functions |

---

## Phase 4: GEMM → Official Doc Mapping

### Weeks 17-24
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Shared memory tiling | Ch. 5.4: Shared Memory | Ch. 5.3: Shared Memory |
| Register pressure | Ch. 5.3.2: Register Memory | Ch. 10.3: Register Pressure |
| Tensor cores | Ch. 7.24: WMMA | — |
| Mixed precision | — | Ch. 13: Math Libraries |

---

## Phase 5: DL Kernels → Official Doc Mapping

### Weeks 25-34
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Warp reductions | Ch. 7.22: Warp Shuffle | — |
| Online algorithms | — | — |
| Memory traffic | — | Ch. 5.1: Memory Throughput |

---

## Phase 6: ML Stack → Official Doc Mapping

### Weeks 35-42
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| CUDA Runtime API | Ch. B: Runtime API | — |
| Streams | Ch. 3.2.7: Streams | Ch. 6.1: Streams |
| Events | Ch. 3.2.8: Events | — |

---

## Phase 7: Multi-GPU → Official Doc Mapping

### Weeks 43-48
| Topic | Programming Guide Section | Best Practices Section |
|-------|---------------------------|------------------------|
| Multi-device | Ch. 3.2.6: Multi-Device | Ch. 11: Multi-GPU |
| Peer access | Ch. 3.2.6.4: Peer Access | — |
| CUDA Graphs | Ch. 3.2.9: Graphs | Ch. 6.3: CUDA Graphs |

---

## Quick Reference Links

### Primary (Always Open)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Profiling (Phase 0+)
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)

### Libraries (Phase 3+)
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/)
- [CUB](https://nvlabs.github.io/cub/)
- [CUTLASS](https://github.com/NVIDIA/cutlass)

### Advanced (Phase 6+)
- [PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [Triton](https://triton-lang.org/)

---

## Daily Practice Template

```markdown
## Date: YYYY-MM-DD

### Official Doc Reading (30 min)
- Section read: ___
- Key insight: ___

### Coding Session
- Kernel/feature: ___
- Doc reference used: ___

### Verification
- Did profiler confirm understanding? Y/N
- Gap discovered: ___
```

---

> **The difference between a good CUDA developer and a great one:**
> Great developers read the official docs weekly. They find optimizations
> others miss because they understand not just *how* to use features,
> but *why* the hardware works the way it does.
