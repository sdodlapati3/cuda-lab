# Distributed Training Benchmarks

Comprehensive benchmarking suite for distributed training performance.

## Overview

This module provides benchmarks for:
- **Communication** - NCCL collective operations bandwidth
- **Training** - End-to-end training throughput
- **Memory** - GPU memory usage patterns
- **Scaling** - Multi-GPU/node efficiency

## Quick Start

```bash
# Run all benchmarks on 4 GPUs
torchrun --nproc_per_node=4 distributed_benchmarks.py

# Run specific benchmark
torchrun --nproc_per_node=4 distributed_benchmarks.py --benchmark comm

# Custom configuration
torchrun --nproc_per_node=8 distributed_benchmarks.py \
    --warmup 20 \
    --iters 200 \
    --output-dir ./results
```

## Benchmarks

### 1. Communication Benchmarks

Measures NCCL collective operation bandwidth:

| Operation | What it measures |
|-----------|------------------|
| All-Reduce | Ring all-reduce bandwidth |
| All-Gather | Gather from all ranks |
| Reduce-Scatter | Reduce and distribute |
| Point-to-Point | Direct GPU-to-GPU transfer |

**Expected Results (H100 NVLink):**
- All-Reduce: ~450 GB/s (bidirectional)
- All-Gather: ~400 GB/s
- P2P: ~100 GB/s per link

### 2. Training Benchmarks

End-to-end training throughput:
- Samples per second
- Step time (ms)
- Memory usage
- Scaling efficiency

### 3. Memory Benchmarks

GPU memory profiling:
- Parameter memory
- Gradient memory
- Activation memory per batch size

## Output Format

Results are saved as JSON:

```json
{
  "name": "benchmark_4gpu",
  "world_size": 4,
  "timestamp": "2024-01-15T10:30:00",
  "config": {
    "warmup_iters": 10,
    "benchmark_iters": 100,
    "device": "NVIDIA H100"
  },
  "results": {
    "communication": {
      "allreduce": {...},
      "allgather": {...}
    },
    "training": {...},
    "memory": {...}
  }
}
```

## Understanding Results

### Communication Bandwidth

For all-reduce, the bus bandwidth formula is:
```
bus_bandwidth = data_size * 2 * (n-1) / n / time
```

Where n is the number of GPUs.

### Scaling Efficiency

Calculate scaling efficiency:
```python
efficiency = (throughput_N_gpus / throughput_1_gpu) / N * 100
```

Good scaling: >80% efficiency
Excellent scaling: >90% efficiency

## Troubleshooting

### Low Bandwidth

1. Check NCCL topology:
```bash
# Set NCCL debug
export NCCL_DEBUG=INFO
```

2. Verify NVLink:
```bash
nvidia-smi topo -m
```

3. Use optimal algorithms:
```bash
export NCCL_ALGO=Ring  # or Tree
```

### Training Bottlenecks

1. Profile with Nsight:
```bash
nsys profile torchrun ...
```

2. Check data loading:
```python
# Add timing around data loading
```

3. Verify overlapping:
```python
# Ensure communication overlaps with compute
```
