# CUDA Learning Path

> 🎯 **Interactive, hands-on CUDA learning through Jupyter notebooks**

## 🚀 Getting Started

**First time?** See **[SETUP-GPU.md](SETUP-GPU.md)** for how to access a T4 GPU via:
- Google Colab (free, easiest)
- ODU HPC cluster
- Cloud providers (AWS, Lambda Labs)

This is your primary learning resource - structured week-by-week with interactive notebooks that combine theory, code examples, and exercises in one place.

## 📚 Structure

Each week contains:
- **Daily notebooks** - Theory + guided coding + exercises
- **Exercise folders** - Independent practice problems
- **Checkpoint quiz** - Self-assessment before moving on

## 🗓️ 12-Week MVP Curriculum

| Week | Focus | Key Skills |
|------|-------|------------|
| **1** | GPU Fundamentals | Device query, first kernel, thread indexing |
| **2** | Memory Basics | cudaMalloc, cudaMemcpy, error handling |
| **3** | Parallel Patterns I | Vector operations, grid-stride loops |
| **4** | Reduction | Sum, min/max, warp-level primitives |
| **5** | Scan | Prefix sum, stream compaction |
| **6** | Matrix Operations | GEMM naive → tiled → optimized |
| **7** | Memory Optimization | Coalescing, shared memory, bank conflicts |
| **8** | Profiling | Nsight Compute, roofline analysis |
| **9** | Streams & Concurrency | Async execution, overlap |
| **10** | Advanced Patterns | Histograms, sorting, atomics |
| **11** | Multi-GPU & Dynamic Parallelism | Scaling up |
| **12** | Capstone Project | End-to-end optimized application |

## 🚀 Getting Started

### Prerequisites
- CUDA Toolkit installed (`nvcc --version`)
- Jupyter with CUDA support
- Basic C/C++ knowledge

### Start Learning

```bash
cd learning-path/week-01
jupyter notebook day-1-gpu-basics.ipynb
```

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

## 📁 Directory Layout

```
learning-path/
├── README.md                 # This file
├── week-01/
│   ├── day-1-gpu-basics.ipynb
│   ├── day-2-first-kernel.ipynb
│   ├── day-3-thread-indexing.ipynb
│   ├── day-4-memory-basics.ipynb
│   ├── exercises/
│   │   ├── ex-device-query/
│   │   └── ex-vector-add/
│   └── checkpoint-quiz.md
├── week-02/
│   └── ...
└── ...
```

## 🔗 Related Resources

- [CUDA Programming Guide](../cuda-programming-guide/) - Reference documentation
- [Quick Reference](../notes/cuda-quick-reference.md) - Cheatsheet
- [Practice Exercises](../practice/) - Additional exercises

---

*Start with Week 1 and progress sequentially. Each week builds on previous knowledge.*
