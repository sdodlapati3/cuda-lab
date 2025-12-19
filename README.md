# CUDA Lab 🚀

A personal CUDA programming learning repository with documentation and practice examples.

## 📜 Learning Philosophy

> **CUDA C++ First, Python/Numba as Optional Backup**

This repository prioritizes **real CUDA C++ programming**:
- All notebooks show **CUDA C++ code first** as the primary learning material
- **Python/Numba** code is provided as an **optional alternative** for quick interactive testing in Colab
- The goal is to learn **actual CUDA programming**, not just GPU abstractions
- Practice exercises use `.cu` files compiled with `nvcc`

**Why this approach?**
- CUDA C++ is the industry standard for GPU programming
- Understanding real CUDA gives you transferable skills to any framework
- Python/Numba is useful for prototyping but hides important details
- Most production CUDA code is written in C++

---

## 🚀 Quick Access

| Resource | Description |
|----------|-------------|
| 🎯 **[START HERE: Learning Path](learning-path/README.md)** | **Interactive notebooks for learning CUDA** |
| 📅 **[12-Week Curriculum](learning-path/12-week-curriculum.md)** | Structured learning plan |
| 📖 **[CUDA Programming Guide](cuda-programming-guide/index.md)** | Full reference documentation |
| ⚡ **[Quick Reference Cheatsheet](notes/cuda-quick-reference.md)** | Common patterns & syntax |
| 🔬 **[Practice Exercises](practice/)** | Standalone CUDA C++ exercises |

---

## 🎓 Learning Path (Recommended)

The **[Learning Path](learning-path/README.md)** provides interactive Jupyter notebooks that combine theory, code examples, and exercises in one place.

### Week 1: GPU Fundamentals (Available Now!)
| Day | Notebook | Topics |
|-----|----------|--------|
| 1 | [GPU Basics](learning-path/week-01/day-1-gpu-basics.ipynb) | CPU vs GPU, device query, first kernel |
| 2 | [Thread Indexing](learning-path/week-01/day-2-thread-indexing.ipynb) | 1D/2D indexing, grid-stride loops |
| 3 | [Memory Basics](learning-path/week-01/day-3-memory-basics.ipynb) | Transfers, pinned memory, optimization |
| 4 | [Error Handling](learning-path/week-01/day-4-error-handling.ipynb) | Debugging, common pitfalls |
| 5 | [Checkpoint Quiz](learning-path/week-01/checkpoint-quiz.md) | Self-assessment |

See the **[12-Week Curriculum](learning-path/12-week-curriculum.md)** for the complete plan.

---

## 📚 CUDA Programming Guide

**30 markdown files** (21,500+ lines) from [NVIDIA CUDA Programming Guide v13.1](https://docs.nvidia.com/cuda/cuda-programming-guide/) (December 2025).

| Section | Files | Topics |
|---------|-------|--------|
| [01-introduction](cuda-programming-guide/01-introduction/) | 3 | CUDA platform, programming model, hardware |
| [02-basics](cuda-programming-guide/02-basics/) | 6 | Intro to CUDA C++, kernels, memory, streams, nvcc |
| [03-advanced](cuda-programming-guide/03-advanced/) | 5 | Performance optimization, memory access, driver API |
| [04-special-topics](cuda-programming-guide/04-special-topics/) | 11 | Unified memory, graphs, cooperative groups, dynamic parallelism |
| [05-appendices](cuda-programming-guide/05-appendices/) | 5 | Compute capabilities, C++ extensions, environment vars |

### 🎯 Key Topics (Quick Links)

| Topic | File |
|-------|------|
| Getting Started | [intro-to-cuda-cpp.md](cuda-programming-guide/02-basics/intro-to-cuda-cpp.md) |
| Writing Kernels | [writing-cuda-kernels.md](cuda-programming-guide/02-basics/writing-cuda-kernels.md) |
| Memory Hierarchy | [understanding-memory.md](cuda-programming-guide/02-basics/understanding-memory.md) |
| Async & Streams | [asynchronous-execution.md](cuda-programming-guide/02-basics/asynchronous-execution.md) |
| Performance | [performance-optimization.md](cuda-programming-guide/03-advanced/performance-optimization.md) |
| Unified Memory | [unified-memory.md](cuda-programming-guide/04-special-topics/unified-memory.md) |
| CUDA Graphs | [cuda-graphs.md](cuda-programming-guide/04-special-topics/cuda-graphs.md) |
| Cooperative Groups | [cooperative-groups.md](cuda-programming-guide/04-special-topics/cooperative-groups.md) |
| C++ Extensions | [cpp-language-extensions.md](cuda-programming-guide/05-appendices/cpp-language-extensions.md) |
| Compute Caps | [compute-capabilities.md](cuda-programming-guide/05-appendices/compute-capabilities.md) |

---

## 📝 Notes

Your own study notes and practice plans.

| File | Description |
|------|-------------|
| [cuda-quick-reference.md](notes/cuda-quick-reference.md) | Cheatsheet with common CUDA patterns |

---

## 🔬 Practice

Hands-on CUDA programming examples and experiments.

```
practice/
├── hello-cuda/          # Basic CUDA setup and first kernels
├── memory-examples/     # Memory management patterns
├── async-examples/      # Streams, events, async execution
└── kernel-examples/     # Various kernel implementations
```

---

## 🛠️ Setup

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit 13.1+
- nvcc compiler

### Verify Installation
```bash
nvcc --version
nvidia-smi
```

### View Documentation (Optional)
```bash
# Install MkDocs for web-style navigation
pip install mkdocs mkdocs-material

# Serve locally at http://localhost:8000
mkdocs serve
```

---

## 🎯 Learning Path

1. [ ] Part 1: Introduction to CUDA
2. [ ] Part 2: Programming GPUs in CUDA  
3. [ ] Part 3: Advanced CUDA
4. [ ] Part 4: CUDA Features
5. [ ] Part 5: Technical Appendices

---
*Happy parallel programming! 🎮*
