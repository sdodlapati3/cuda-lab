# Roofline Analysis Tools

Tools for generating roofline model visualizations to understand kernel performance characteristics.

## Overview

The roofline model is a visual performance model that helps identify whether a kernel is:
- **Memory-bound**: Performance limited by memory bandwidth
- **Compute-bound**: Performance limited by compute throughput

## Files

| File | Purpose |
|------|---------|
| `plot_roofline.py` | Generate roofline visualizations |
| `extract_ncu_metrics.py` | Extract metrics from Nsight Compute reports |

## Quick Start

### 1. Generate a Basic Roofline

```bash
module load python3

# Plot with example kernels
crun -p ~/envs/cuda-lab python plot_roofline.py --examples --output roofline_examples.png

# Specify hardware (A100, H100, V100, T4, RTX-4090)
crun -p ~/envs/cuda-lab python plot_roofline.py --hardware H100-SXM --examples
```

### 2. Profile Your Kernel

```bash
# Run Nsight Compute on your CUDA executable
ncu --csv --metrics dram__bytes_read.sum,dram__bytes_write.sum,\
smsp__sass_thread_inst_executed_op_fp32_pred_on.sum,gpu__time_duration.avg \
-o kernel_report ./your_kernel

# Or use our extraction script
crun -p ~/envs/cuda-lab python extract_ncu_metrics.py --run ./your_kernel --output my_kernels.json
```

### 3. Plot Your Results

```bash
# Generate roofline with your kernel data
crun -p ~/envs/cuda-lab python plot_roofline.py --kernels my_kernels.json --output my_roofline.png
```

## Understanding the Roofline

```
                    ▲ Performance (TFLOPS)
                    │
    Peak Compute →  │─────────────────────────
                    │           ╱
                    │          ╱  Compute-bound
                    │         ╱   region
                    │        ╱
                    │       ╱
                    │      ╱  Memory-bound
                    │     ╱   region
                    │    ╱
                    └────┼────────────────────▶
                         │    Arithmetic Intensity
                    Ridge Point    (FLOPs/Byte)
```

### Key Concepts

- **Arithmetic Intensity (AI)**: FLOPs per byte of memory traffic
  - Low AI (< 10): Memory-bound (e.g., vector add, reduction)
  - High AI (> 100): Compute-bound (e.g., dense matrix multiply)

- **Ridge Point**: Where memory roof meets compute roof
  - AI = Peak TFLOPS / Peak Bandwidth
  - A100: ~9.6 FLOPs/Byte
  - H100: ~20 FLOPs/Byte

- **Kernel Efficiency**: Achieved / Theoretical maximum at that AI

## Hardware Profiles

Built-in profiles for common GPUs:

| GPU | FP32 TFLOPS | Bandwidth | Ridge Point |
|-----|-------------|-----------|-------------|
| A100-80GB | 19.5 | 2039 GB/s | 9.6 |
| H100-SXM | 67 | 3350 GB/s | 20.0 |
| V100 | 15.7 | 900 GB/s | 17.4 |
| T4 | 8.1 | 320 GB/s | 25.3 |
| RTX-4090 | 82.6 | 1008 GB/s | 81.9 |

### Custom Hardware

```bash
python plot_roofline.py --peak-tflops 30.0 --peak-bw 1500
```

## JSON Kernel Format

```json
{
  "kernels": [
    {
      "name": "My Kernel",
      "arithmetic_intensity": 12.5,
      "performance": 15.2,
      "color": "blue",
      "marker": "o"
    }
  ]
}
```

## Common Kernel Arithmetic Intensities

| Operation | Typical AI | Classification |
|-----------|-----------|----------------|
| Vector Add | 0.08 | Memory-bound |
| Reduction | 0.25 | Memory-bound |
| Softmax | 2-5 | Memory-bound |
| GEMV | 1-2 | Memory-bound |
| Convolution | 10-100 | Transitional |
| GEMM (naive) | 10-30 | Transitional |
| GEMM (tiled) | 50-200 | Compute-bound |
| FlashAttention | 100+ | Compute-bound |

## Integration with Profiling Workflow

```bash
# 1. Profile with Nsight Compute
cd ../profiling-lab/02-nsight-compute
# Follow exercises to collect metrics

# 2. Extract metrics
python ../../benchmarks/roofline/extract_ncu_metrics.py --csv report.csv

# 3. Generate roofline
python ../../benchmarks/roofline/plot_roofline.py --kernels kernels.json
```

## References

- [Williams et al., "Roofline: An Insightful Visual Performance Model"](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [GPU Roofline Model](https://developer.nvidia.com/blog/using-the-roofline-model-to-guide-optimizations-of-gpu-applications/)
