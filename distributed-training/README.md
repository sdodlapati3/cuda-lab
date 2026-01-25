# Distributed Training Guide

A comprehensive guide for distributed deep learning across multi-GPU and multi-node configurations.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Module Structure](#module-structure)
4. [Quick Start](#quick-start)
5. [Choosing a Strategy](#choosing-a-strategy)
6. [Performance Tips](#performance-tips)

---

## Overview

This module covers the essential distributed training strategies:

| Strategy | Best For | Memory Efficiency | Communication |
|----------|----------|-------------------|---------------|
| **DDP** | Models that fit in GPU memory | Low | All-reduce gradients |
| **FSDP** | Large models, memory constrained | High | All-gather params + reduce-scatter grads |
| **DeepSpeed ZeRO** | Very large models, flexible | Very High | Configurable sharding |
| **Tensor Parallelism** | Huge layers (attention, FFN) | Medium | Point-to-point |
| **Pipeline Parallelism** | Very deep models | Medium | Pipeline bubbles |

### Parallelism Types

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARALLELISM STRATEGIES                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DATA PARALLELISM (DDP/FSDP/ZeRO)                                  │
│  ─────────────────────────────────                                  │
│  • Same model replicated across GPUs                                │
│  • Different data batches per GPU                                   │
│  • Gradients synchronized via all-reduce                            │
│                                                                     │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                                │
│  │Model│  │Model│  │Model│  │Model│  ← Same weights                │
│  │Copy │  │Copy │  │Copy │  │Copy │                                │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                                │
│     │        │        │        │                                    │
│  Batch 1  Batch 2  Batch 3  Batch 4  ← Different data              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MODEL PARALLELISM (Tensor/Pipeline)                               │
│  ───────────────────────────────────                                │
│  • Model split across GPUs                                          │
│  • Same data flows through distributed model                        │
│                                                                     │
│  Tensor Parallel:     Pipeline Parallel:                           │
│  ┌─────┬─────┐        ┌─────┐                                      │
│  │Attn │Attn │        │Layer│──┐                                   │
│  │Left │Right│        │ 1-4 │  │                                   │
│  └─────┴─────┘        └─────┘  │                                   │
│       GPU 0,1               GPU 0                                   │
│                            ┌─────┐                                  │
│  ┌─────┬─────┐             │Layer│◄─┘                              │
│  │FFN  │FFN  │             │ 5-8 │                                  │
│  │Left │Right│             └─────┘                                  │
│  └─────┴─────┘               GPU 1                                  │
│       GPU 0,1                                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

```bash
# Core requirements
pip install torch torchvision
pip install deepspeed
pip install fairscale  # Optional: for advanced FSDP features

# Verify multi-GPU setup
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Verify NCCL
python -c "import torch.distributed as dist; print('NCCL available:', dist.is_nccl_available())"
```

---

## Module Structure

```
distributed-training/
├── README.md                    # This file
├── 01-ddp-basics/               # DistributedDataParallel
│   ├── README.md
│   ├── ddp_mnist.py             # Simple end-to-end example
│   ├── ddp_training.py          # Production-ready template
│   ├── ddp_benchmark.py         # Scaling efficiency test
│   └── launch_ddp.sh            # Launch scripts
│
├── 02-fsdp/                     # Fully Sharded Data Parallel
│   ├── README.md
│   ├── fsdp_basics.py           # Basic FSDP usage
│   ├── fsdp_transformer.py      # Training large transformers
│   ├── fsdp_checkpoint.py       # Checkpointing strategies
│   └── sharding_strategies.md   # When to use what
│
├── 03-deepspeed/                # DeepSpeed ZeRO
│   ├── README.md
│   ├── configs/                 # ZeRO stage configurations
│   │   ├── zero1_config.json
│   │   ├── zero2_config.json
│   │   └── zero3_config.json
│   ├── deepspeed_train.py       # Complete training script
│   ├── zero_offload.py          # CPU/NVMe offloading
│   └── deepspeed_inference.py   # Inference with DeepSpeed
│
├── 04-nccl-operations/          # NCCL collective operations
│   ├── README.md
│   ├── collective_ops.py        # all_reduce, all_gather, etc.
│   ├── point_to_point.py        # send/recv operations
│   ├── bandwidth_test.py        # Inter-GPU bandwidth
│   └── nccl_debugging.md        # Troubleshooting guide
│
├── 05-advanced/                 # Advanced patterns
│   ├── README.md
│   ├── hybrid_parallelism.py    # Combining strategies
│   ├── gradient_accumulation.py # Effective batch scaling
│   └── communication_overlap.py # Hide communication latency
│
└── benchmarks/                  # Performance benchmarks
    ├── README.md
    └── distributed_benchmarks.py # Communication & training benchmarks
```

---

## Quick Start

### 1. Single Node Multi-GPU (DDP)

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 01-ddp-basics/ddp_mnist.py

# Using SLURM
srun --gpus=4 --ntasks=4 python 01-ddp-basics/ddp_mnist.py
```

### 2. Multi-Node Training

```bash
# Node 0 (master)
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 \
    01-ddp-basics/ddp_training.py

# Node 1
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=1 --master_addr=10.0.0.1 --master_port=29500 \
    01-ddp-basics/ddp_training.py
```

### 3. DeepSpeed Training

```bash
deepspeed --num_gpus=4 03-deepspeed/deepspeed_train.py \
    --deepspeed_config 03-deepspeed/configs/zero2_config.json
```

---

## Choosing a Strategy

```
                    ┌─────────────────────────────────────┐
                    │    Does model fit in GPU memory?    │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │                                     │
                   YES                                   NO
                    │                                     │
                    ▼                                     ▼
            ┌───────────────┐               ┌─────────────────────────┐
            │   Use DDP     │               │  Is it a transformer?   │
            │  (Simplest)   │               └────────────┬────────────┘
            └───────────────┘                            │
                                         ┌───────────────┴───────────────┐
                                         │                               │
                                        YES                             NO
                                         │                               │
                                         ▼                               ▼
                              ┌─────────────────────┐        ┌─────────────────┐
                              │  Use FSDP or ZeRO-3 │        │ Use Pipeline or │
                              │  (Shard everything) │        │ Tensor Parallel │
                              └─────────────────────┘        └─────────────────┘
```

### Decision Matrix

| Scenario | Recommended Strategy | Why |
|----------|---------------------|-----|
| ResNet-50, 8 GPUs | DDP | Model fits, simple setup |
| GPT-2 (1.5B), 4 GPUs | FSDP FULL_SHARD | Moderate size, needs sharding |
| LLaMA-7B, 8 GPUs | DeepSpeed ZeRO-3 | Large model, need offloading |
| LLaMA-70B, 64 GPUs | 3D Parallelism | Very large, need all strategies |
| Fine-tuning LLM | FSDP + LoRA | Memory efficient adaptation |

---

## Performance Tips

### 1. Gradient Accumulation for Effective Batch Size

```python
# Effective batch = micro_batch * accumulation_steps * world_size
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Mixed Precision (Always Use!)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. Communication Optimization

```python
# Enable gradient bucketing (DDP default, tune bucket size)
model = DDP(model, bucket_cap_mb=25)

# Use NCCL for GPU, Gloo for CPU
dist.init_process_group(backend='nccl')
```

### 4. Profile Before Optimizing

```bash
# Profile distributed training
nsys profile --trace=cuda,nvtx,osrt \
    torchrun --nproc_per_node=4 train.py
```

---

## Environment Variables

```bash
# NCCL Configuration for GCP Waterfield Cluster (no InfiniBand)
export NCCL_DEBUG=INFO                    # Debug output (use WARN in production)
export NCCL_IB_DISABLE=1                  # GCP has no InfiniBand
export NCCL_SOCKET_IFNAME=eth0            # Use Ethernet interface
export NCCL_SOCKET_NTHREADS=4             # More threads for socket ops
export NCCL_NSOCKS_PERTHREAD=2            # Multiple sockets per thread
export NCCL_BUFFSIZE=4194304              # 4MB buffer for throughput

# For clusters WITH InfiniBand (AWS p4d, Azure ND, etc.):
# export NCCL_IB_DISABLE=0                # Enable InfiniBand
# export NCCL_NET_GDR_LEVEL=2             # GPUDirect RDMA
# export NCCL_SOCKET_IFNAME=ib0           # InfiniBand interface

# PyTorch Distributed
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0
export LOCAL_RANK=0

# Performance
export OMP_NUM_THREADS=4                  # CPU threads per process
export CUDA_DEVICE_MAX_CONNECTIONS=1      # For overlap optimization
```

---

## Next Steps

1. Start with [01-ddp-basics](01-ddp-basics/README.md) for fundamentals
2. Move to [02-fsdp](02-fsdp/README.md) for large model training
3. Explore [03-deepspeed](03-deepspeed/README.md) for production scale
4. Learn [04-nccl-operations](04-nccl-operations/README.md) for communication optimization

---

*Last updated: January 2026*
