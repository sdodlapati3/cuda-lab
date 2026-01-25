# DeepSpeed Integration Module

This module provides comprehensive examples for training with Microsoft DeepSpeed,
focusing on ZeRO optimization stages and efficient large model training.

## Overview

DeepSpeed is a deep learning optimization library that enables:
- **ZeRO (Zero Redundancy Optimizer)**: Memory-efficient distributed training
- **3D Parallelism**: Data + Tensor + Pipeline parallelism
- **Mixed Precision**: FP16/BF16 training with loss scaling
- **Gradient Compression**: Reduced communication overhead
- **CPU/NVMe Offloading**: Train models larger than GPU memory

## ZeRO Optimization Stages

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ZeRO MEMORY OPTIMIZATION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ZeRO Stage 0 (DDP Baseline):                                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │                   │
│  ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤                   │
│  │ Params  │ │ Params  │ │ Params  │ │ Params  │  Full copy        │
│  │ Grads   │ │ Grads   │ │ Grads   │ │ Grads   │  on each GPU      │
│  │ Optim   │ │ Optim   │ │ Optim   │ │ Optim   │                   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
│                                                                      │
│  ZeRO Stage 1 (Optimizer State Partitioning):                       │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │                   │
│  ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤                   │
│  │ Params  │ │ Params  │ │ Params  │ │ Params  │                   │
│  │ Grads   │ │ Grads   │ │ Grads   │ │ Grads   │                   │
│  │ Optim/4 │ │ Optim/4 │ │ Optim/4 │ │ Optim/4 │  4x reduction     │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
│                                                                      │
│  ZeRO Stage 2 (+ Gradient Partitioning):                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │                   │
│  ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤                   │
│  │ Params  │ │ Params  │ │ Params  │ │ Params  │                   │
│  │ Grads/4 │ │ Grads/4 │ │ Grads/4 │ │ Grads/4 │  8x reduction     │
│  │ Optim/4 │ │ Optim/4 │ │ Optim/4 │ │ Optim/4 │                   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
│                                                                      │
│  ZeRO Stage 3 (+ Parameter Partitioning):                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │                   │
│  ├─────────┤ ├─────────┤ ├─────────┤ ├─────────┤                   │
│  │Params/4 │ │Params/4 │ │Params/4 │ │Params/4 │  Linear scaling   │
│  │ Grads/4 │ │ Grads/4 │ │ Grads/4 │ │ Grads/4 │  with # GPUs      │
│  │ Optim/4 │ │ Optim/4 │ │ Optim/4 │ │ Optim/4 │                   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                   │
│                                                                      │
│  ZeRO-Infinity (+ CPU/NVMe Offloading):                             │
│  ┌─────────┐ ┌─────────┐       ┌──────────────────┐                │
│  │  GPU    │ │  GPU    │ ←──── │   CPU Memory     │                │
│  │ Active  │ │ Active  │       │   Offloaded      │                │
│  │ Params  │ │ Params  │       │   Params/Optim   │                │
│  └─────────┘ └─────────┘       └────────┬─────────┘                │
│                                          │                          │
│                                    ┌─────▼─────┐                   │
│                                    │   NVMe    │                   │
│                                    │  Storage  │                   │
│                                    └───────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Memory Savings by Stage

| Stage | Optimizer States | Gradients | Parameters | Memory vs DDP |
|-------|-----------------|-----------|------------|---------------|
| ZeRO-0 | Full | Full | Full | 1x (baseline) |
| ZeRO-1 | Partitioned | Full | Full | 4x reduction |
| ZeRO-2 | Partitioned | Partitioned | Full | 8x reduction |
| ZeRO-3 | Partitioned | Partitioned | Partitioned | Linear with GPUs |
| ZeRO-Infinity | Offloaded | Offloaded | Offloaded | Beyond GPU memory |

## When to Use Each Stage

```
┌────────────────────────────────────────────────────────────────────┐
│                    DEEPSPEED DECISION TREE                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Model fits in single GPU?                                         │
│  ├── YES → Use DDP (fastest)                                       │
│  └── NO → Continue...                                              │
│                                                                     │
│  Model fits with ZeRO-1?                                           │
│  ├── YES → Use ZeRO-1 (minimal overhead)                           │
│  └── NO → Continue...                                              │
│                                                                     │
│  Model fits with ZeRO-2?                                           │
│  ├── YES → Use ZeRO-2 (good balance)                               │
│  └── NO → Continue...                                              │
│                                                                     │
│  Model fits with ZeRO-3?                                           │
│  ├── YES → Use ZeRO-3 (max memory efficiency)                      │
│  └── NO → Continue...                                              │
│                                                                     │
│  Have enough CPU memory?                                           │
│  ├── YES → Use ZeRO-Offload (CPU)                                  │
│  └── NO → Use ZeRO-Infinity (NVMe)                                 │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Module Contents

### Configuration Files
- `zero1_config.json` - ZeRO Stage 1 configuration
- `zero2_config.json` - ZeRO Stage 2 configuration  
- `zero3_config.json` - ZeRO Stage 3 configuration
- `zero_offload_config.json` - ZeRO-Offload configuration

### Training Scripts
- `deepspeed_train.py` - Basic DeepSpeed training example
- `deepspeed_hf.py` - HuggingFace integration
- `zero_offload.py` - CPU/NVMe offloading example

## Quick Start

### Installation

```bash
pip install deepspeed

# Optional: For advanced features
pip install deepspeed[sparse_attn]
pip install deepspeed[1bit_adam]
```

### Basic Training

```bash
# Single node, 4 GPUs with ZeRO-2
deepspeed --num_gpus=4 deepspeed_train.py \
    --deepspeed_config zero2_config.json

# Multi-node training
deepspeed --hostfile hostfile.txt --num_gpus=4 deepspeed_train.py \
    --deepspeed_config zero3_config.json
```

### Configuration Selection Guide

```python
# ZeRO Stage Selection
def select_zero_stage(model_params_billions, gpu_memory_gb, num_gpus):
    """Select appropriate ZeRO stage."""
    
    # Memory per GPU with different stages (rough estimates)
    # Assumes AdamW optimizer (12 bytes per param for optim states)
    
    bytes_per_param = 2  # BF16
    optim_bytes = 12     # AdamW states in FP32
    
    # Model memory
    model_memory = model_params_billions * 1e9 * bytes_per_param / 1e9  # GB
    optim_memory = model_params_billions * 1e9 * optim_bytes / 1e9      # GB
    
    # Memory per GPU
    zero0_per_gpu = model_memory + optim_memory + model_memory  # params + optim + grads
    zero1_per_gpu = model_memory + (optim_memory / num_gpus) + model_memory
    zero2_per_gpu = model_memory + (optim_memory / num_gpus) + (model_memory / num_gpus)
    zero3_per_gpu = (model_memory + optim_memory + model_memory) / num_gpus
    
    # Select stage
    if zero0_per_gpu < gpu_memory_gb * 0.7:  # 70% threshold
        return 0, "Use DDP"
    elif zero1_per_gpu < gpu_memory_gb * 0.7:
        return 1, "Use ZeRO-1"
    elif zero2_per_gpu < gpu_memory_gb * 0.7:
        return 2, "Use ZeRO-2"
    elif zero3_per_gpu < gpu_memory_gb * 0.7:
        return 3, "Use ZeRO-3"
    else:
        return 3, "Use ZeRO-3 with CPU offload"
```

## Performance Considerations

### Communication Overhead

| Stage | Forward | Backward | Step | Total Overhead |
|-------|---------|----------|------|----------------|
| ZeRO-1 | - | - | All-gather | Low |
| ZeRO-2 | - | Reduce-scatter | All-gather | Medium |
| ZeRO-3 | All-gather | Reduce-scatter | All-gather | High |

### Optimization Tips

1. **Overlap Communication**: Enable `overlap_comm` in config
2. **Gradient Bucketing**: Tune `reduce_bucket_size` 
3. **Prefetching**: Enable `stage3_prefetch_bucket_size`
4. **Contiguous Gradients**: Set `contiguous_gradients: true`

## Integration with Other Features

### Mixed Precision
```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    }
}
```

### Gradient Accumulation
```json
{
    "gradient_accumulation_steps": 4,
    "train_micro_batch_size_per_gpu": 4
}
```

### Activation Checkpointing
```python
import deepspeed
deepspeed.checkpointing.configure(
    mp_size=1,
    enabled=True,
    num_checkpoints=num_layers
)
```

## Resources

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [ZeRO-Infinity Paper](https://arxiv.org/abs/2104.07857)
- [DeepSpeed Examples](https://github.com/microsoft/DeepSpeedExamples)
