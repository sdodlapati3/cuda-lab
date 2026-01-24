# Nsight Compute - Kernel-Level Analysis

> **Deep-dive into kernel performance metrics and roofline analysis**

Nsight Compute is NVIDIA's kernel-level profiler. While Nsight Systems shows the big picture, Nsight Compute shows you **exactly why** a kernel is slow.

---

## 🎯 Learning Objectives

After completing these exercises, you will be able to:
- Measure memory bandwidth utilization
- Calculate achieved vs theoretical occupancy
- Plot kernels on a roofline diagram
- Identify memory-bound vs compute-bound kernels

---

## 🔧 Tool Setup

### Installation
Nsight Compute comes with CUDA Toolkit:
```bash
ncu --version
```

### Basic Usage
```bash
# Profile all kernels
ncu ./my_cuda_app

# Profile specific kernel
ncu --kernel-name "vector_add" ./my_cuda_app

# Save full report
ncu -o report ./my_cuda_app

# Collect specific metrics
ncu --set full -o report ./my_cuda_app
```

### Opening Reports
```bash
# Launch GUI
ncu-ui report.ncu-rep

# Print summary
ncu --import report.ncu-rep --print-summary per-kernel
```

---

## 📚 Exercises

| Exercise | Topic | Time | Difficulty |
|----------|-------|------|------------|
| [ex01-memory-metrics](ex01-memory-metrics/) | Bandwidth, cache hit rates | 1.5 hr | ⭐⭐⭐ |
| [ex02-compute-metrics](ex02-compute-metrics/) | Occupancy, warp efficiency | 1.5 hr | ⭐⭐⭐ |
| [ex03-roofline-practice](ex03-roofline-practice/) | Roofline analysis | 2 hr | ⭐⭐⭐⭐ |
| [ex04-optimization-loop](ex04-optimization-loop/) | Profile-optimize-reprofile | 2 hr | ⭐⭐⭐⭐ |

---

## 🔑 Key Metrics

### Memory Metrics

| Metric | What It Measures | Target |
|--------|------------------|--------|
| `dram__bytes.sum` | Total DRAM bytes accessed | - |
| `dram__throughput.avg.pct_of_peak_sustained` | DRAM bandwidth utilization | >80% |
| `l2_cache_hit_rate` | L2 cache effectiveness | >50% |
| `sm__sass_data_bytes_mem_global_op_ld.sum` | Global memory loads | - |

### Compute Metrics

| Metric | What It Measures | Target |
|--------|------------------|--------|
| `sm__throughput.avg.pct_of_peak_sustained` | SM utilization | >70% |
| `sm__warps_active.avg.pct_of_peak_sustained` | Achieved occupancy | 50-100% |
| `sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained` | FP32 utilization | >70% |
| `sm__sass_thread_inst_executed_op_ffma_pred_on.avg` | FFMA (fused multiply-add) | - |

### Efficiency Metrics

| Metric | What It Measures | Target |
|--------|------------------|--------|
| `smsp__warp_issue_stalled_*` | Stall reasons | Minimize |
| `sm__warps_active.avg.per_cycle_active` | Active warps per cycle | Higher = better |
| `launch__occupancy_limit_*` | What limits occupancy | - |

---

## 📊 Roofline Model

### Concept
The roofline model helps you understand if a kernel is **memory-bound** or **compute-bound**.

```
Performance (GFLOPS)
        ^
        |         * Compute Ceiling
        |        /
        |       /
        |      /  <- Ridge point
        |     /
        |    /  * Memory Ceiling
        |   /
        +--+----------------------> Arithmetic Intensity (FLOPS/Byte)
```

### Calculating Arithmetic Intensity
```
AI = FLOPS / Bytes Accessed

Example: Vector Add
- Each element: 1 FLOP (addition)
- Each element: 12 bytes (read a, read b, write c)
- AI = 1 / 12 = 0.083 FLOPS/Byte → Memory-bound!

Example: Matrix Multiply (tiled)
- Each element: ~2*N FLOPS
- Each element: ~2*N/tile_size bytes
- AI = tile_size FLOPS/Byte → Can be compute-bound with large tiles
```

### Using Nsight Compute Roofline
```bash
# Collect roofline data
ncu --set roofline -o roofline_report ./my_cuda_app

# View in GUI
ncu-ui roofline_report.ncu-rep
# Navigate to "Roofline" section
```

---

## 📋 Quick Reference

### Common ncu Commands
```bash
# Quick summary
ncu --target-processes all ./app

# Full metrics
ncu --set full -o report ./app

# Specific section
ncu --section SpeedOfLight -o report ./app

# Memory workload analysis
ncu --section MemoryWorkloadAnalysis -o report ./app

# Source correlation (needs -lineinfo)
ncu --set source -o report ./app
```

### Metric Sets
| Set | Purpose |
|-----|---------|
| `default` | Basic performance overview |
| `full` | All available metrics |
| `roofline` | Roofline analysis data |
| `source` | Source-level correlation |
| `memory_workload_analysis` | Detailed memory analysis |

### Interpreting Speedof Light
The "Speed of Light" section shows how close you are to hardware limits:

| SoL % | Interpretation |
|-------|----------------|
| >80% | Excellent |
| 60-80% | Good |
| 40-60% | Room for improvement |
| <40% | Significant issues |

---

## 🔧 Compilation for Profiling

```bash
# Include line info for source correlation
nvcc -O3 -lineinfo -arch=sm_80 kernel.cu -o kernel

# For detailed register analysis
nvcc -O3 -lineinfo --ptxas-options=-v -arch=sm_80 kernel.cu -o kernel
```

---

## 📖 Further Reading

- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [Roofline Model Paper](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
