# CUDA Programming Guide

> 📖 **Source:** [NVIDIA CUDA Programming Guide v13.1](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) (December 2025)

---

## 📚 Table of Contents

### 1. Introduction
| # | Topic | Description |
|---|-------|-------------|
| 1.1 | [CUDA Platform](01-introduction/cuda-platform.md) | Compute capability, toolkit, driver, runtime |
| 1.2 | [Programming Model](01-introduction/programming-model.md) | Kernels, thread hierarchy, memory hierarchy |
| 1.3 | [Hardware Implementation](01-introduction/hardware-implementation.md) | SIMT architecture, hardware multithreading |

### 2. Basics
| # | Topic | Description |
|---|-------|-------------|
| 2.1 | [Intro to CUDA C++](02-basics/intro-to-cuda-cpp.md) | First kernels, compilation, runtime initialization |
| 2.2 | [Writing CUDA Kernels](02-basics/writing-cuda-kernels.md) | Thread indexing, shared memory, synchronization |
| 2.3 | [CUDA Memory](02-basics/cuda-memory.md) | Memory allocation, transfers, pinned memory |
| 2.4 | [Understanding Memory](02-basics/understanding-memory.md) | Memory types, coalescing, caching |
| 2.5 | [Asynchronous Execution](02-basics/asynchronous-execution.md) | Streams, events, concurrent execution |
| 2.6 | [NVCC Compiler](02-basics/nvcc.md) | Compilation, PTX, separate compilation |

### 3. Advanced
| # | Topic | Description |
|---|-------|-------------|
| 3.1 | [Advanced Kernel Programming](03-advanced/advanced-kernel-programming.md) | Warp-level, thread scopes, launch bounds |
| 3.2 | [Advanced Host Programming](03-advanced/advanced-host-programming.md) | Context management, synchronization |
| 3.3 | [Device Memory Access](03-advanced/device-memory-access.md) | Coalescing, alignment, memory patterns |
| 3.4 | [Performance Optimization](03-advanced/performance-optimization.md) | Occupancy, latency hiding, profiling |
| 3.5 | [Driver API](03-advanced/driver-api.md) | Low-level API, modules, contexts |

### 4. Special Topics
| # | Topic | Description |
|---|-------|-------------|
| 4.1 | [Unified Memory](04-special-topics/unified-memory.md) | Managed memory, page migration, hints |
| 4.2 | [Multi-GPU Programming](04-special-topics/multi-gpu-programming.md) | Peer access, multi-device |
| 4.3 | [CUDA Graphs](04-special-topics/cuda-graphs.md) | Graph creation, capture, execution |
| 4.4 | [Dynamic Parallelism](04-special-topics/dynamic-parallelism.md) | Device-side kernel launches |
| 4.5 | [Cooperative Groups](04-special-topics/cooperative-groups.md) | Flexible thread groups, synchronization |
| 4.6 | [Stream-Ordered Memory](04-special-topics/stream-ordered-memory-allocation.md) | Async allocation, memory pools |
| 4.7 | [Virtual Memory Management](04-special-topics/virtual-memory-management.md) | Virtual address spaces, mapping |
| 4.8 | [Inter-Process Communication](04-special-topics/inter-process-communication.md) | IPC memory handles |
| 4.9 | [Programmatic Dependent Launch](04-special-topics/programmatic-dependent-launch.md) | Kernel dependencies |
| 4.10 | [Error & Log Management](04-special-topics/error-log-management.md) | Error handling, logging |
| 4.11 | [MIG](04-special-topics/mig.md) | Multi-Instance GPU |

### 5. Appendices
| # | Topic | Description |
|---|-------|-------------|
| 5.1 | [C++ Language Extensions](05-appendices/cpp-language-extensions.md) | `__device__`, `__global__`, built-in variables |
| 5.2 | [Compute Capabilities](05-appendices/compute-capabilities.md) | Feature tables by compute capability |
| 5.3 | [Warp Matrix Functions](05-appendices/warp-matrix-functions.md) | Tensor Core WMMA operations |
| 5.4 | [Texture Fetch](05-appendices/texture-fetch.md) | Texture memory operations |
| 5.5 | [Environment Variables](05-appendices/environment-variables.md) | CUDA runtime environment vars |

---

## 🚀 Quick Start Path

New to CUDA? Follow this learning path:

1. **[Programming Model](01-introduction/programming-model.md)** - Understand the basics
2. **[Intro to CUDA C++](02-basics/intro-to-cuda-cpp.md)** - Write your first kernel
3. **[Writing CUDA Kernels](02-basics/writing-cuda-kernels.md)** - Thread indexing & shared memory
4. **[Understanding Memory](02-basics/understanding-memory.md)** - Memory hierarchy
5. **[Performance Optimization](03-advanced/performance-optimization.md)** - Make it fast!

---

## 🔗 Quick Reference

→ **[CUDA Quick Reference Cheatsheet](../notes/cuda-quick-reference.md)** - Common patterns & syntax
