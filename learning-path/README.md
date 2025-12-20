# 🚀 CUDA Learning Path - 16 Week Curriculum

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-01/day-1-gpu-basics.ipynb)

> 🎯 **Interactive, hands-on CUDA learning through Jupyter notebooks**

A comprehensive 16-week CUDA programming curriculum with hands-on Jupyter notebooks. All notebooks are designed to run on Google Colab with T4 GPU.

> **⚠️ Before running in Colab:** Go to `Runtime → Change runtime type → T4 GPU`!

---

## 📖 Learning Philosophy

> **CUDA C++ First, Python/Numba as Optional Backup**

Every notebook follows this pattern:
1. **Concept explanation** with diagrams and theory
2. **CUDA C++ code** as the PRIMARY implementation (executable via `%%writefile` + `!nvcc`)
3. **Python/Numba code** as OPTIONAL for quick interactive testing

**Why CUDA C++ First?**
- CUDA C++ is what you'll use in production
- Understanding real CUDA gives transferable skills
- Python/Numba hides memory management and other crucial details
- The goal is to become a CUDA programmer, not just use GPU abstractions

---

## 📚 Curriculum Overview

| Week | Focus Area | Topics Covered |
|------|------------|----------------|
| 1 | GPU Fundamentals | GPU architecture, thread indexing, memory basics, error handling |
| 2 | Memory Hierarchy | Coalescing, shared memory, bank conflicts, constant/texture memory |
| 3 | Parallel Patterns I | Grid-stride loops, elementwise ops, SAXPY, fused operations |
| 4 | Parallel Patterns II | Reduction, warp primitives, atomics, histogram |
| 5 | Scan Algorithms | Inclusive/exclusive scan, Hillis-Steele, Blelloch, large arrays |
| 6 | Matrix Operations | Naive/tiled matmul, transpose optimization, cuBLAS |
| 7 | Occupancy & Resources | Occupancy tuning, register pressure, cache optimization |
| 8 | Profiling | Nsight Compute, roofline analysis, Nsight Systems |
| 9 | Streams | Stream basics, overlapping transfers, multi-stream, events |
| 10 | CUDA Graphs | Graph basics, explicit API, updates, optimization |
| 11 | Advanced Features | Cooperative groups, grid sync, dynamic parallelism |
| 12 | Multi-GPU & Capstone | Multi-GPU programming, optimization review, capstone project |
| 13 | Unified Memory | Managed memory, prefetching, migration, oversubscription |
| 14 | Memory Management | Virtual memory, pools, async allocation, fragmentation |
| 15 | Advanced Sync | Grid-wide sync, programmatic launch, cooperative kernels |
| 16 | Final Capstone | Integration project, real-world optimization, best practices |

---

## 🗺️ Learning Roadmap

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                   CUDA LEARNING PATH                      │
                    └──────────────────────────────────────────────────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────┐
         │                                    │                                │
         ▼                                    ▼                                ▼
┌─────────────────┐               ┌─────────────────┐              ┌─────────────────┐
│  FOUNDATIONS    │               │   OPTIMIZATION   │              │   ADVANCED      │
│   Weeks 1-6     │───────────────│   Weeks 7-12     │──────────────│   Weeks 13-16   │
└─────────────────┘               └─────────────────┘              └─────────────────┘
         │                                    │                                │
         ▼                                    ▼                                ▼
• GPU Architecture         • Occupancy Tuning          • Unified Memory
• Thread/Block Model       • Profiling Tools           • Virtual Memory
• Memory Hierarchy         • Streams & Events          • Advanced Sync
• Parallel Patterns        • CUDA Graphs               • Real-world Projects
• Scan & Reduction         • Multi-GPU Basics          • Production Patterns
• Matrix Operations        • Capstone Project          • Final Integration
```

### Skill Progression

| Phase | Weeks | Skills Acquired | Practice Exercises |
|-------|-------|-----------------|-------------------|
| **Foundations** | 1-6 | Write kernels, manage memory, implement algorithms | [01-foundations](../practice/01-foundations/), [02-memory](../practice/02-memory/) |
| **Optimization** | 7-12 | Profile, tune, use streams/graphs, multi-GPU | [03-parallel](../practice/03-parallel/), [04-optimization](../practice/04-optimization/) |
| **Advanced** | 13-16 | Unified memory, advanced sync, production code | [05-advanced](../practice/05-advanced/) |

---

## 🎯 Week 1: GPU Fundamentals

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | GPU Basics & First Kernel | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-01/day-1-gpu-basics.ipynb) |
| 2 | Thread Indexing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-01/day-2-thread-indexing.ipynb) |
| 3 | Memory Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-01/day-3-memory-basics.ipynb) |
| 4 | Error Handling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-01/day-4-error-handling.ipynb) |

---

## 🧠 Week 2: Memory Hierarchy

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Memory Coalescing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-02/day-1-memory-coalescing.ipynb) |
| 2 | Shared Memory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-02/day-2-shared-memory.ipynb) |
| 3 | Bank Conflicts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-02/day-3-bank-conflicts.ipynb) |
| 4 | Special Memory Types | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-02/day-4-special-memory.ipynb) |

---

## ⚡ Week 3: Parallel Patterns I

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Grid-Stride Loops | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-03/day-1-grid-stride-loops.ipynb) |
| 2 | Elementwise Operations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-03/day-2-elementwise-ops.ipynb) |
| 3 | SAXPY & BLAS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-03/day-3-saxpy-blas.ipynb) |
| 4 | Fused Operations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-03/day-4-fused-operations.ipynb) |

---

## 🔄 Week 4: Parallel Patterns II

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Parallel Reduction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-04/day-1-parallel-reduction.ipynb) |
| 2 | Warp Primitives | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-04/day-2-warp-primitives.ipynb) |
| 3 | Atomic Operations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-04/day-3-atomic-operations.ipynb) |
| 4 | Histogram | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-04/day-4-histogram.ipynb) |

---

## 📊 Week 5: Scan Algorithms

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Scan Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-05/day-1-scan-basics.ipynb) |
| 2 | Hillis-Steele Scan | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-05/day-2-hillis-steele.ipynb) |
| 3 | Blelloch Scan | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-05/day-3-blelloch.ipynb) |
| 4 | Large Array Scan | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-05/day-4-large-scan.ipynb) |

---

## 🔢 Week 6: Matrix Operations

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Naive Matrix Multiply | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-06/day-1-naive-matmul.ipynb) |
| 2 | Tiled Matrix Multiply | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-06/day-2-tiled-matmul.ipynb) |
| 3 | Matrix Transpose | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-06/day-3-transpose.ipynb) |
| 4 | cuBLAS Library | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-06/day-4-cublas.ipynb) |

---

## ⚙️ Week 7: Occupancy & Resources

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Occupancy Optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-07/day-1-occupancy.ipynb) |
| 2 | Register Pressure | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-07/day-2-registers.ipynb) |
| 3 | Cache Optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-07/day-3-cache.ipynb) |

---

## 📈 Week 8: Profiling & Analysis

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Nsight Compute | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-08/day-1-nsight-compute.ipynb) |
| 2 | Roofline Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-08/day-2-roofline.ipynb) |
| 3 | Nsight Systems | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-08/day-3-nsight-systems.ipynb) |
| 4 | Optimization Case Study | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-08/day-4-optimization-case-study.ipynb) |

---

## 🌊 Week 9: CUDA Streams

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Stream Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-09/day-1-stream-basics.ipynb) |
| 2 | Overlapping Transfers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-09/day-2-overlap-transfers.ipynb) |
| 3 | Multi-Stream Patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-09/day-3-multi-stream.ipynb) |
| 4 | CUDA Events | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-09/day-4-events.ipynb) |

---

## 📊 Week 10: CUDA Graphs

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Graph Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-10/day-1-graph-basics.ipynb) |
| 2 | Explicit Graph API | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-10/day-2-explicit-graphs.ipynb) |
| 3 | Graph Updates | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-10/day-3-graph-updates.ipynb) |
| 4 | Graph Optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-10/day-4-graph-optimization.ipynb) |

---

## 🔬 Week 11: Advanced Features

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Cooperative Groups | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-11/day-1-cooperative-groups.ipynb) |
| 2 | Grid Synchronization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-11/day-2-grid-sync.ipynb) |
| 3 | Dynamic Parallelism | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-11/day-3-dynamic-parallelism.ipynb) |

---

## 🏆 Week 12: Multi-GPU & Capstone

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Multi-GPU Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-12/day-1-multi-gpu-basics.ipynb) |
| 2 | Multi-GPU Patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-12/day-2-multi-gpu-patterns.ipynb) |
| 3 | Optimization Review | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-12/day-3-optimization-review.ipynb) |
| 4 | Capstone Project | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-12/day-4-capstone.ipynb) |

---

## 🧮 Week 13: Unified Memory

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Managed Memory Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-13/day-1-managed-memory.ipynb) |
| 2 | Prefetching & Hints | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-13/day-2-prefetching.ipynb) |
| 3 | Memory Migration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-13/day-3-migration.ipynb) |
| 4 | Oversubscription | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-13/day-4-oversubscription.ipynb) |

---

## 💾 Week 14: Memory Management

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Virtual Memory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-14/day-1-virtual-memory.ipynb) |
| 2 | Memory Pools | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-14/day-2-memory-pools.ipynb) |
| 3 | Async Allocation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-14/day-3-async-allocation.ipynb) |
| 4 | Fragmentation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-14/day-4-fragmentation.ipynb) |

---

## 🔄 Week 15: Advanced Synchronization

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Grid-Wide Sync | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-15/day-1-grid-sync.ipynb) |
| 2 | Programmatic Launch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-15/day-2-programmatic-launch.ipynb) |
| 3 | Cooperative Kernels | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-15/day-3-cooperative-kernels.ipynb) |
| 4 | Sync Patterns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-15/day-4-sync-patterns.ipynb) |

---

## 🎓 Week 16: Final Capstone

| Day | Topic | Open in Colab |
|-----|-------|---------------|
| 1 | Integration Project | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-16/day-1-integration.ipynb) |
| 2 | Real-World Optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-16/day-2-real-world.ipynb) |
| 3 | Best Practices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-16/day-3-best-practices.ipynb) |
| 4 | Final Project | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-16/day-4-final-project.ipynb) |

---

## 🛠️ Setup Instructions

### Google Colab (Recommended)
1. Click any "Open in Colab" badge above
2. Go to `Runtime → Change runtime type`
3. Select `T4 GPU` as hardware accelerator
4. Click `Save`
5. Run all cells!

### Local Setup
```bash
# Clone the repository
git clone https://github.com/sdodlapati3/cuda-lab.git
cd cuda-lab/learning-path

# Ensure CUDA toolkit is installed
nvcc --version

# Run Jupyter
jupyter notebook
```

**First time?** See **[SETUP-GPU.md](SETUP-GPU.md)** for detailed GPU access options.

---

## 📊 Progress Tracking

| Week | Status | Completed Date | Notes |
|------|--------|----------------|-------|
| Week 1 | ⬜ Not Started | | GPU Fundamentals |
| Week 2 | ⬜ Not Started | | Memory Hierarchy |
| Week 3 | ⬜ Not Started | | Parallel Patterns I |
| Week 4 | ⬜ Not Started | | Parallel Patterns II |
| Week 5 | ⬜ Not Started | | Scan Algorithms |
| Week 6 | ⬜ Not Started | | Matrix Operations |
| Week 7 | ⬜ Not Started | | Occupancy & Resources |
| Week 8 | ⬜ Not Started | | Profiling & Analysis |
| Week 9 | ⬜ Not Started | | CUDA Streams |
| Week 10 | ⬜ Not Started | | CUDA Graphs |
| Week 11 | ⬜ Not Started | | Advanced Features |
| Week 12 | ⬜ Not Started | | Multi-GPU & Capstone |
| Week 13 | ⬜ Not Started | | Unified Memory |
| Week 14 | ⬜ Not Started | | Memory Management |
| Week 15 | ⬜ Not Started | | Advanced Sync |
| Week 16 | ⬜ Not Started | | Final Capstone |

---

## 🛠️ Practice Exercises

Hands-on coding exercises to reinforce each week's concepts:

| Directory | Focus | Exercises |
|-----------|-------|-----------|
| [01-foundations](../practice/01-foundations/) | Weeks 1-3 | Device query, Hello GPU |
| [02-memory](../practice/02-memory/) | Week 2 | Coalescing, Shared memory, Bank conflicts, Transpose |
| [03-parallel](../practice/03-parallel/) | Weeks 4-5 | Reduction, Warp primitives, Scan, Histogram |
| [04-optimization](../practice/04-optimization/) | Weeks 7-9 | Occupancy, Streams, Events |
| [05-advanced](../practice/05-advanced/) | Weeks 10-16 | Graphs, Unified memory, Integration |

---

## 🔗 Related Resources

- [CUDA Programming Guide](../cuda-programming-guide/) - Reference documentation
- [Quick Reference](../notes/cuda-quick-reference.md) - Cheatsheet
- [Practice Exercises](../practice/) - Additional exercises

---

## 🎓 Prerequisites

- Basic C/C++ programming knowledge
- Understanding of parallel computing concepts (helpful but not required)
- No prior GPU programming experience needed!

---

*Start with Week 1 and progress sequentially. Each week builds on previous knowledge. Happy CUDA Learning! 🚀*
