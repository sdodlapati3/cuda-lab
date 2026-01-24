# PyTorch Optimization Track

Advanced PyTorch performance optimization techniques for production ML systems.

## Overview

This module covers GPU-aware optimization patterns that go beyond basic PyTorch usage:

- **torch.compile** - PyTorch 2.0 JIT compilation
- **Mixed Precision** - AMP and explicit FP16/BF16
- **Operator Fusion** - Fused kernels for common patterns
- **Memory Optimization** - Gradient checkpointing, activation compression
- **Distributed Optimization** - FSDP, tensor parallelism

## Module Structure

```
pytorch-optimization/
├── README.md
├── 01-torch-compile/          # PyTorch 2.0 compilation
├── 02-mixed-precision/        # AMP and precision strategies
├── 03-fused-operations/       # Operator fusion patterns
├── 04-memory-efficiency/      # Memory optimization
└── 05-distributed/            # Distributed training patterns
```

## Quick Reference

### torch.compile

```python
# Basic compilation
model = torch.compile(model)

# With mode selection
model = torch.compile(model, mode="reduce-overhead")  # Low latency
model = torch.compile(model, mode="max-autotune")     # Maximum throughput

# Disable for debugging
model = torch.compile(model, mode="reduce-overhead", disable=True)
```

### Mixed Precision

```python
# Automatic Mixed Precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Fused Operations

```python
# Fused Adam optimizer
optimizer = torch.optim.AdamW(model.parameters(), fused=True)

# Fused layer norm
from torch.nn.functional import layer_norm
# PyTorch auto-fuses when beneficial

# Flash Attention (PyTorch 2.0+)
from torch.nn.functional import scaled_dot_product_attention
```

## Performance Guidelines

| Optimization | When to Use | Expected Speedup |
|--------------|-------------|------------------|
| torch.compile | Always (PT 2.0+) | 1.3-2x |
| AMP (FP16) | Training & inference | 1.5-3x |
| Fused optimizers | Large models | 1.1-1.3x |
| Flash Attention | Transformers | 2-4x |
| Gradient checkpointing | Memory-limited | 0.8x speed, 3x memory |

## Getting Started

```bash
# Ensure environment is set up
module load python3

# Run torch.compile benchmarks
crun -p ~/envs/cuda-lab python 01-torch-compile/compile_benchmark.py

# Run mixed precision examples
crun -p ~/envs/cuda-lab python 02-mixed-precision/amp_training.py
```
