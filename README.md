# CUDA Lab 🚀

A personal CUDA programming learning repository with documentation and practice examples.

## 📚 Documentation

**30 markdown files** (21,500+ lines) from NVIDIA CUDA Programming Guide v13.1 (December 2025).

| Section | Files | Topics |
|---------|-------|--------|
| [01-introduction](docs/01-introduction/) | 3 | CUDA platform, programming model, hardware |
| [02-basics](docs/02-basics/) | 6 | Intro to CUDA C++, kernels, memory, streams, nvcc |
| [03-advanced](docs/03-advanced/) | 5 | Performance optimization, memory access, driver API |
| [04-special-topics](docs/04-special-topics/) | 11 | Unified memory, graphs, cooperative groups, dynamic parallelism |
| [05-appendices](docs/05-appendices/) | 5 | Compute capabilities, C++ extensions, environment vars |

### Key Documentation Files

| Topic | File |
|-------|------|
| Getting Started | [intro-to-cuda-cpp.md](docs/02-basics/intro-to-cuda-cpp.md) |
| Writing Kernels | [writing-cuda-kernels.md](docs/02-basics/writing-cuda-kernels.md) |
| Memory Hierarchy | [understanding-memory.md](docs/02-basics/understanding-memory.md) |
| Async & Streams | [asynchronous-execution.md](docs/02-basics/asynchronous-execution.md) |
| Performance | [performance-optimization.md](docs/03-advanced/performance-optimization.md) |
| Unified Memory | [unified-memory.md](docs/04-special-topics/unified-memory.md) |
| CUDA Graphs | [cuda-graphs.md](docs/04-special-topics/cuda-graphs.md) |
| Cooperative Groups | [cooperative-groups.md](docs/04-special-topics/cooperative-groups.md) |
| Compute Caps | [compute-capabilities.md](docs/05-appendices/compute-capabilities.md) |

## 🔬 Practice

Hands-on CUDA programming examples and experiments.

```
practice/
├── hello-cuda/          # Basic CUDA setup and first kernels
├── memory-examples/     # Memory management patterns
├── async-examples/      # Streams, events, async execution
└── kernel-examples/     # Various kernel implementations
```

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

## 📖 Quick Reference

| Concept | Location |
|---------|----------|
| Unified Memory | `docs/04-special-topics/unified-memory.md` |
| CUDA Graphs | `docs/04-special-topics/cuda-graphs.md` |
| Cooperative Groups | `docs/04-special-topics/cooperative-groups.md` |
| Compute Capabilities | `docs/05-appendices/compute-capabilities.md` |

## 📝 Notes

- Documentation sourced from [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- Last synced: December 2025 (CUDA 13.1)

## 🎯 Learning Path

1. [ ] Part 1: Introduction to CUDA
2. [ ] Part 2: Programming GPUs in CUDA
3. [ ] Part 3: Advanced CUDA
4. [ ] Part 4: CUDA Features
5. [ ] Part 5: Technical Appendices

---
*Happy parallel programming! 🎮*
