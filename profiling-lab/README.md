# 📊 Profiling Lab

> **Master GPU performance analysis with industry-standard tools**

This lab develops the profiling skills essential for HPC and ML performance engineering roles. After completing this module, you'll be able to:

- Identify memory-bound vs compute-bound kernels using roofline analysis
- Diagnose scaling efficiency drops in distributed training
- Make concrete performance decisions backed by profiler evidence
- Measure and optimize energy efficiency

---

## 📈 Current Status

| Module | Documentation | Source Code | Profile Reports | Status |
|--------|---------------|-------------|-----------------|--------|
| 01-nsight-systems | ✅ Complete | ✅ `baseline.cu`, `improved.cu` | ✅ 2 `.nsys-rep` files | **Ready** |
| 02-nsight-compute | ✅ Complete | ✅ 4 CUDA programs | ⬜ Not yet generated | **Ready** |
| 03-pytorch-profiler | ✅ Complete | ✅ `profile_training.py` | ⬜ Not yet generated | Partial |
| 04-energy-profiling | ✅ Complete | ✅ `energy_benchmark.py` | ⬜ Not yet generated | Partial |
| 05-advanced-nsight | ✅ Complete | ✅ 8 Python files | ⬜ Not yet generated | **Ready** |

**Last Updated:** January 2026

---

## 🎯 NESAP Skill Alignment

| NESAP Requirement | This Lab Covers |
|-------------------|-----------------|
| "Performance bugs often come from hardware misunderstandings" | Nsight Compute kernel analysis |
| "NESAP success = measured improvements, not guesses" | Profiling-driven optimization workflow |
| "Can you reason about when scaling efficiency drops?" | Distributed training profiling |
| "Science per joule" | Energy profiling module |

---

## 📚 Modules

### [01-nsight-systems/](01-nsight-systems/)
**Timeline Analysis & System-Level Profiling**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-timeline-basics | Read timeline, identify GPU idle time | ⭐⭐ |
| ex02-kernel-overlap | Streams and kernel concurrency | ⭐⭐⭐ |
| ex03-memory-timeline | H2D/D2H transfer analysis | ⭐⭐⭐ |
| ex04-multi-gpu-timeline | NCCL communication profiling | ⭐⭐⭐⭐ |

**Key skills:** CPU-GPU overlap, kernel launch latency, communication bottlenecks

---

### [02-nsight-compute/](02-nsight-compute/)
**Kernel-Level Analysis & Roofline**

| Exercise | Topic | Source Code | Difficulty |
|----------|-------|-------------|------------|
| ex01-memory-metrics | Bandwidth utilization, cache hit rates | `memory_bandwidth.cu` | ⭐⭐⭐ |
| ex02-compute-metrics | Occupancy, warp execution efficiency | `compute_metrics.cu` | ⭐⭐⭐ |
| ex03-roofline-practice | Plot kernels, identify bottleneck type | `roofline_kernels.cu` | ⭐⭐⭐⭐ |
| ex04-optimization-loop | Profile → optimize → reprofile cycle | `optimization_loop.cu` | ⭐⭐⭐⭐ |

**Key skills:** Memory-bound vs compute-bound, optimization targeting

**Quick Start:**
```bash
cd 02-nsight-compute/ex01-memory-metrics
make && ./memory_bandwidth
ncu --section MemoryWorkloadAnalysis ./memory_bandwidth
```

---

### [03-pytorch-profiler/](03-pytorch-profiler/)
**PyTorch Profiler & TensorBoard Integration**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-basic-profiling | torch.profiler context manager | ⭐⭐ |
| ex02-memory-profiling | Memory snapshots, allocation tracking | ⭐⭐⭐ |
| ex03-distributed-profiling | DDP training profiling | ⭐⭐⭐⭐ |

**Key skills:** Python-level profiling, memory debugging, distributed analysis

---

### [04-energy-profiling/](04-energy-profiling/)
**Power Monitoring & Energy Efficiency**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-power-measurement | nvidia-smi, NVML APIs | ⭐⭐ |
| ex02-energy-efficiency | TFLOPS/Watt, energy-to-solution | ⭐⭐⭐ |

**Key skills:** Power monitoring, efficiency metrics for HPC

---

### [05-scaling-benchmarks/](05-scaling-benchmarks/)
**Strong & Weak Scaling Analysis**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-single-gpu-baseline | Establish baseline metrics | ⭐⭐ |
| ex02-multi-gpu-scaling | Strong scaling efficiency | ⭐⭐⭐⭐ |
| ex03-communication-overhead | Isolate communication costs | ⭐⭐⭐⭐ |

**Key skills:** Scaling efficiency, communication analysis, benchmark design

---

### [05-advanced-nsight/](05-advanced-nsight/)
**Advanced Nsight Systems Features**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-python-backtrace | Python/C++ call stack correlation | ⭐⭐⭐ |
| ex02-io-dataloader | I/O & DataLoader bottleneck profiling | ⭐⭐⭐ |
| ex03-nsys-stats-cli | CLI-based analysis without GUI | ⭐⭐⭐ |
| ex04-sqlite-analysis | SQLite export & custom queries | ⭐⭐⭐⭐ |
| ex05-cpu-sampling | CPU sampling for hotspot detection | ⭐⭐⭐ |
| ex06-osrt-tracing | OS runtime tracing (syscalls, I/O) | ⭐⭐⭐⭐ |
| ex07-comparison-reports | Before/after optimization comparison | ⭐⭐⭐⭐ |
| ex08-expert-systems | Custom expert rules & auto-analysis | ⭐⭐⭐⭐⭐ |

**Key skills:** Python stack correlation, scriptable profiling, automated analysis, CI/CD integration

---

## 🔧 Tool Installation

### Nsight Systems
```bash
# Usually bundled with CUDA Toolkit
nsys --version

# Or install separately
# https://developer.nvidia.com/nsight-systems
```

### Nsight Compute
```bash
# Usually bundled with CUDA Toolkit
ncu --version

# May require sudo or admin rights for full metrics
```

### PyTorch Profiler
```bash
pip install torch-tb-profiler  # TensorBoard plugin
```

---

## 📋 Prerequisites

- Completed learning-path Weeks 1-8 (or equivalent)
- Access to NVIDIA GPU (T4 minimum, A100 recommended)
- For distributed profiling: Multi-GPU system or HPC cluster

---

## 🎯 Learning Outcomes

After completing this lab, you should be able to:

1. **Explain** whether a kernel is memory-bound or compute-bound with evidence
2. **Identify** the root cause of scaling efficiency loss
3. **Demonstrate** a profiling-driven optimization (before/after metrics)
4. **Design** a fair benchmark for kernel comparison
5. **Measure** energy efficiency for HPC workloads

---

## 📊 Key Metrics Cheatsheet

| Metric | Tool | Good Value | Indicates |
|--------|------|------------|-----------|
| GPU Utilization | nvidia-smi, nsys | >90% | GPU not idle |
| Memory Bandwidth | ncu | >80% of peak | Memory efficient |
| Compute Throughput | ncu | >70% of peak | Compute efficient |
| Occupancy | ncu | 50-100% | Depends on kernel |
| Scaling Efficiency | custom | >90% at 4 GPU | Good parallelization |

---

## �️ Profiling Utilities

This lab includes ready-to-use profiling tools in the `utils/` directory:

| Utility | Description | Use Case |
|---------|-------------|----------|
| [unified_profiler.py](utils/unified_profiler.py) | All-in-one profiling for any GPU configuration | Single/Multi-GPU/Multi-Node/DeepSpeed |
| [easy_profiler.py](utils/easy_profiler.py) | Minimal-code decorators and context managers | Quick benchmarks and prototyping |

### Quick Start

```bash
# Profile single GPU training
python utils/unified_profiler.py train.py

# Profile multi-GPU training
python utils/unified_profiler.py --mode multi-gpu --gpus 4 train.py

# Profile DeepSpeed training
nsys profile -o deepspeed_trace deepspeed --num_gpus=4 train.py
```

### In Your Code

```python
from profiling_lab.utils.easy_profiler import auto_profile, GPUMonitor

# Automatic profiling with decorator
@auto_profile(warmup=3, iterations=10)
def train_step(model, batch):
    return model(batch)

# GPU monitoring context manager
with GPUMonitor() as monitor:
    for batch in dataloader:
        train_step(batch)
print(monitor.get_summary())
```

---

## 📖 Comprehensive Guides

| Guide | Description |
|-------|-------------|
| [COMPREHENSIVE-PROFILING-GUIDE.md](COMPREHENSIVE-PROFILING-GUIDE.md) | Complete profiling guide: single GPU to multi-node, DeepSpeed, FSDP |
| [PROFILER-COMPARISON.md](PROFILER-COMPARISON.md) | Tool comparison and decision matrix |

---

## 📚 Reference Materials

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [NVIDIA Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [DeepSpeed FLOPs Profiler](https://www.deepspeed.ai/tutorials/flops-profiler/)
- [Roofline Model Paper](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)

---

*Start with [01-nsight-systems/](01-nsight-systems/) for timeline analysis, or jump to the [COMPREHENSIVE-PROFILING-GUIDE.md](COMPREHENSIVE-PROFILING-GUIDE.md) for distributed/DeepSpeed profiling.*
