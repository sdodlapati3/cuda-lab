# 🚀 CUDA Learning Path - 12 Week Curriculum

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sdodlapati3/cuda-lab/blob/main/learning-path/week-01/day-1-gpu-basics.ipynb)

> 🎯 **Interactive, hands-on CUDA learning through Jupyter notebooks**

A comprehensive 12-week CUDA programming curriculum with hands-on Jupyter notebooks. All notebooks are designed to run on Google Colab with T4 GPU.

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
| Week 1 | ⬜ Not Started | | |
| Week 2 | ⬜ Not Started | | |
| Week 3 | ⬜ Not Started | | |
| Week 4 | ⬜ Not Started | | |
| Week 5 | ⬜ Not Started | | |
| Week 6 | ⬜ Not Started | | |
| Week 7 | ⬜ Not Started | | |
| Week 8 | ⬜ Not Started | | |
| Week 9 | ⬜ Not Started | | |
| Week 10 | ⬜ Not Started | | |
| Week 11 | ⬜ Not Started | | |
| Week 12 | ⬜ Not Started | | |

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
