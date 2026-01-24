# CUDA Lab 🚀

A comprehensive CUDA programming learning repository for ML engineers and HPC practitioners.

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

## 🗺️ Choose Your Learning Track

See **[LEARNING-TRACKS.md](LEARNING-TRACKS.md)** for detailed guidance.

| Track | Duration | Audience | Start Here |
|-------|----------|----------|------------|
| **Foundation** | 18 weeks | Anyone learning CUDA | [learning-path/](learning-path/README.md) |
| **Mastery** | 52 weeks | Career GPU engineers | [bootcamp/](bootcamp/README.md) |
| **NESAP/HPC** | 12 weeks | Scientific ML, HPC roles | [profiling-lab/](profiling-lab/) *(coming soon)* |

---

## 🚀 Quick Access

| Resource | Description |
|----------|-------------|
| 🎯 **[Learning Path](learning-path/README.md)** | 18-week interactive notebooks (Colab-compatible) |
| 🏋️ **[Bootcamp](bootcamp/README.md)** | 52-week intensive mastery curriculum |
| 📖 **[CUDA Programming Guide](cuda-programming-guide/index.md)** | Full reference documentation (v13.1) |
| ⚡ **[Quick Reference](notes/cuda-quick-reference.md)** | Common patterns & syntax cheatsheet |
| 🔬 **[Practice Exercises](practice/)** | Standalone CUDA C++ exercises |
| 📊 **[Profiling Lab](profiling-lab/)** | Nsight Systems/Compute mastery *(coming soon)* |
| 🖥️ **[HPC Lab](hpc-lab/)** | Slurm, checkpointing, containers *(coming soon)* |

---

## 🎓 Learning Path (Recommended)

---

## 🎓 Learning Path (Foundation Track)

The **[Learning Path](learning-path/README.md)** provides interactive Jupyter notebooks that combine theory, code examples, and exercises in one place. **18 weeks**, Colab-compatible.

### Week 1: GPU Fundamentals
| Day | Notebook | Topics |
|-----|----------|--------|
| 1 | [GPU Basics](learning-path/week-01/day-1-gpu-basics.ipynb) | CPU vs GPU, device query, first kernel |
| 2 | [Thread Indexing](learning-path/week-01/day-2-thread-indexing.ipynb) | 1D/2D indexing, grid-stride loops |
| 3 | [Memory Basics](learning-path/week-01/day-3-memory-basics.ipynb) | Transfers, pinned memory, optimization |
| 4 | [Error Handling](learning-path/week-01/day-4-error-handling.ipynb) | Debugging, common pitfalls |
| 5 | [Checkpoint Quiz](learning-path/week-01/checkpoint-quiz.md) | Self-assessment |

See **[18-Week Curriculum](learning-path/README.md)** for the complete plan.

---

## 🏋️ Bootcamp (Mastery Track)

The **[Bootcamp](bootcamp/README.md)** is a **52-week intensive** curriculum for ML engineers committed to becoming GPU performance experts.

**Key outcomes:**
- GEMM at >50% cuBLAS performance
- FlashAttention understanding and implementation
- PyTorch C++/CUDA extensions with autograd
- Multi-GPU scaling with NCCL
- Portfolio-quality capstone projects

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

---

## 🔬 Practice Exercises

```
practice/
├── 01-foundations/      # Device query, first kernels
├── 02-memory/           # Coalescing, shared memory, bank conflicts
├── 03-parallel/         # Reduction, scan, histogram
├── 04-optimization/     # Occupancy, streams, events
├── 05-advanced/         # CUDA graphs, cooperative groups, CDP
└── 06-systems/          # IPC, textures, production patterns
```

---

## 📊 Profiling Lab *(Coming Soon)*

Performance analysis mastery for HPC/ML roles:

```
profiling-lab/
├── 01-nsight-systems/   # Timeline analysis, CPU-GPU overlap
├── 02-nsight-compute/   # Kernel metrics, roofline
├── 03-pytorch-profiler/ # torch.profiler integration
├── 04-energy-profiling/ # Power monitoring, efficiency
└── 05-scaling-benchmarks/  # Strong/weak scaling
```

---

## 🖥️ HPC Lab *(Coming Soon)*

HPC workflows for national lab and cluster environments:

```
hpc-lab/
├── 01-slurm-basics/     # Job scripts, arrays, dependencies
├── 02-checkpointing/    # Fault-tolerant training
├── 03-containers/       # Singularity/Apptainer
├── 04-filesystems/      # Lustre, GPFS, I/O patterns
└── 05-debugging-hpc/    # Multi-node debugging
```

---

## 🛠️ Setup

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit 13.1+
- nvcc compiler

### Environment Setup (ODU HPC / NERSC)

```bash
# Set up dedicated Python environment
./scripts/setup-environment.sh

# Running Python scripts
module load python3
crun -p ~/envs/cuda-lab python script.py

# Running tests
crun -p ~/envs/cuda-lab pytest tests/

# Running Jupyter
crun -p ~/envs/cuda-lab jupyter lab
```

**Tip**: Add alias to your `~/.bashrc`:
```bash
alias cudalab='module load python3 && crun -p ~/envs/cuda-lab'
# Then use: cudalab python script.py
```

### Verify Installation
```bash
nvcc --version
nvidia-smi
```

### View Documentation (Optional)
```bash
# Install MkDocs for web-style navigation
crun -p ~/envs/cuda-lab pip install mkdocs mkdocs-material

# Serve locally at http://localhost:8000
crun -p ~/envs/cuda-lab mkdocs serve
```

---

## 📋 Planning Documents

| Document | Description |
|----------|-------------|
| [LEARNING-TRACKS.md](LEARNING-TRACKS.md) | Choose your learning path |
| [docs/NESAP-ALIGNED-ENHANCEMENT-PLAN.md](docs/NESAP-ALIGNED-ENHANCEMENT-PLAN.md) | HPC/NESAP career alignment plan |
| [docs/CURRICULUM-ENHANCEMENT-PLAN.md](docs/CURRICULUM-ENHANCEMENT-PLAN.md) | Completed curriculum updates |
| [docs/modern-gpu-ecosystem.md](docs/modern-gpu-ecosystem.md) | When to use Triton/cuBLAS/etc. |

---

## 🎯 NESAP/HPC Skill Alignment

This repository targets the following HPC/ML role competencies:

| Skill Area | Coverage | Location |
|------------|----------|----------|
| GPU Architecture | ⭐⭐⭐⭐⭐ | learning-path/, cuda-programming-guide/ |
| Performance Profiling | ⭐⭐⭐⭐ | profiling-lab/ *(building)* |
| Distributed Training | ⭐⭐⭐⭐ | bootcamp/phase8 |
| HPC Workflows | ⭐⭐⭐ | hpc-lab/ *(building)* |
| Scientific ML | ⭐⭐ | scientific-ml/ *(planned)* |
| PyTorch at Scale | ⭐⭐⭐⭐ | bootcamp/phase8 |

See [NESAP-ALIGNED-ENHANCEMENT-PLAN.md](docs/NESAP-ALIGNED-ENHANCEMENT-PLAN.md) for the full skill gap analysis.

---

*Happy parallel programming! 🚀*
