# 📊 Benchmarks Suite

> **Standardized benchmarks for kernel performance and scaling analysis**

This directory contains benchmark infrastructure for:
- Kernel performance comparison (vs cuBLAS, CUB, etc.)
- Scaling efficiency (strong/weak scaling)
- Hardware baselines for reproducibility
- Energy efficiency measurements

---

## 🎯 Purpose

| Goal | This Suite Provides |
|------|---------------------|
| "Can you design a fair scaling benchmark?" | Scaling benchmark templates |
| "Measured improvements, not guesses" | Standardized measurement harnesses |
| "Compare to library implementations" | Baselines for cuBLAS, CUB, cuDNN |
| "Reproducible results" | Hardware-specific baseline JSON files |

---

## 📁 Directory Structure

```
benchmarks/
├── README.md                 # This file
├── kernels/                  # Kernel-level benchmarks
│   ├── reduction/
│   ├── matmul/
│   ├── softmax/
│   └── attention/
├── scaling/                  # Multi-GPU scaling
│   ├── strong-scaling/
│   ├── weak-scaling/
│   └── communication-overhead/
├── hardware-baselines/       # Reference numbers
│   ├── T4.json
│   ├── A100-40GB.json
│   ├── A100-80GB.json
│   └── H100.json
├── roofline/                 # Roofline analysis
│   ├── generate-roofline.py
│   └── reference-plots/
├── energy/                   # Power/efficiency
│   └── power-benchmark.py
└── templates/                # Reusable templates
    ├── benchmark-template.py
    ├── scaling-report-template.md
    └── regression-test.py
```

---

## 🔧 Usage

### Kernel Benchmarks

```bash
module load python3
cd kernels/reduction
crun -p ~/envs/cuda-lab python benchmark.py --implementation naive warp_shuffle cub
```

Output:
```
Implementation      | Bandwidth (GB/s) | % Peak | Time (ms)
--------------------|------------------|--------|----------
naive               | 245.3            | 30.7%  | 4.08
warp_shuffle        | 712.8            | 89.1%  | 1.40
cub                 | 756.2            | 94.5%  | 1.32
```

### Scaling Benchmarks

```bash
cd scaling/strong-scaling
crun -p ~/envs/cuda-lab python benchmark.py --gpus 1 2 4 8 --batch-size 1024
```

Output:
```
GPUs | Throughput (samples/s) | Efficiency | Speedup
-----|------------------------|------------|--------
1    | 1024                   | 100%       | 1.00x
2    | 1980                   | 96.7%      | 1.93x
4    | 3840                   | 93.8%      | 3.75x
8    | 7200                   | 87.9%      | 7.03x
```

### Roofline Analysis

```bash
cd roofline
crun -p ~/envs/cuda-lab python generate-roofline.py --gpu A100 --kernels ../kernels/*/
```

---

## 📋 Hardware Baselines

Each hardware baseline JSON contains:

```json
{
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "compute_capability": "8.0",
  "memory_bandwidth_GB_s": 2039,
  "fp32_tflops": 19.5,
  "fp16_tflops": 312,
  "tf32_tflops": 156,
  "int8_tops": 624,
  "tdp_watts": 400,
  "measured_date": "2025-01-24",
  "benchmarks": {
    "reduction_bandwidth_GB_s": 1856,
    "matmul_tflops_fp16": 280,
    "nccl_allreduce_bandwidth_GB_s": 150
  }
}
```

---

## 📊 Benchmark Templates

### benchmark-template.py

```python
"""
Template for kernel benchmarks.
Copy to your kernel directory and customize.
"""

import torch
import time
import json
from pathlib import Path

class KernelBenchmark:
    def __init__(self, name: str, warmup: int = 10, iterations: int = 100):
        self.name = name
        self.warmup = warmup
        self.iterations = iterations
        self.results = {}
    
    def benchmark(self, fn, *args, **kwargs):
        """Benchmark a kernel function."""
        # Warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        
        # Timed iterations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(self.iterations):
            fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        
        time_ms = start.elapsed_time(end) / self.iterations
        return time_ms
    
    def add_result(self, impl_name: str, time_ms: float, bytes_moved: int = 0, flops: int = 0):
        """Record benchmark result."""
        result = {"time_ms": time_ms}
        if bytes_moved > 0:
            result["bandwidth_GB_s"] = bytes_moved / (time_ms * 1e6)
        if flops > 0:
            result["tflops"] = flops / (time_ms * 1e9)
        self.results[impl_name] = result
    
    def save(self, path: str):
        """Save results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_table(self):
        """Print results as table."""
        # Implementation in full template...
        pass
```

---

## 🎯 Key Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Bandwidth Utilization | `bytes / (time * peak_bandwidth)` | >80% |
| Compute Utilization | `flops / (time * peak_flops)` | >70% |
| Scaling Efficiency | `throughput_N / (throughput_1 * N)` | >90% at 4 GPU |
| Energy Efficiency | `throughput / power` | Maximize |

---

## 📚 Reference

- [Roofline Model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
- [MLPerf Benchmarks](https://mlcommons.org/en/inference-datacenter/)
- [NVIDIA cuBLAS Benchmarks](https://developer.nvidia.com/cublas)

---

*Use these benchmarks to validate your kernel implementations and demonstrate optimization results with evidence.*
