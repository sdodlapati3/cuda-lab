# NCCL Operations Module

This module provides hands-on examples for understanding and optimizing
NVIDIA Collective Communications Library (NCCL) operations in distributed training.

## Overview

NCCL is the backbone of GPU-to-GPU communication in PyTorch distributed training.
Understanding NCCL operations is essential for:

- **Debugging** distributed training issues
- **Optimizing** communication patterns
- **Selecting** the right collective for your use case
- **Understanding** what happens under the hood in DDP/FSDP

## NCCL Collective Operations

```
┌────────────────────────────────────────────────────────────────────────┐
│                     NCCL COLLECTIVE OPERATIONS                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ALL-REDUCE (sum gradients across GPUs)                                │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│  │  1  │ │  2  │ │  3  │ │  4  │  →   │ 10  │ │ 10  │ │ 10  │ │ 10  │ │
│  └─────┘ └─────┘ └─────┘ └─────┘      └─────┘ └─────┘ └─────┘ └─────┘ │
│  GPU 0   GPU 1   GPU 2   GPU 3        GPU 0   GPU 1   GPU 2   GPU 3   │
│  (Each GPU has sum of all values)                                      │
│                                                                         │
│  ALL-GATHER (gather data to all GPUs)                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      ┌───────────────────────┐        │
│  │  A  │ │  B  │ │  C  │ │  D  │  →   │   A   B   C   D       │ × 4   │
│  └─────┘ └─────┘ └─────┘ └─────┘      └───────────────────────┘        │
│  (Each GPU has all pieces)                                             │
│                                                                         │
│  REDUCE-SCATTER (reduce and distribute)                                │
│  ┌───────────────────────┐            ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│  │ A₀ A₁ A₂ A₃          │ × 4   →    │ΣA₀  │ │ΣA₁  │ │ΣA₂  │ │ΣA₃  │ │
│  └───────────────────────┘            └─────┘ └─────┘ └─────┘ └─────┘ │
│  (Each GPU has reduced portion)                                        │
│                                                                         │
│  BROADCAST (one-to-all)                                                │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│  │  X  │ │     │ │     │ │     │  →   │  X  │ │  X  │ │  X  │ │  X  │ │
│  └─────┘ └─────┘ └─────┘ └─────┘      └─────┘ └─────┘ └─────┘ └─────┘ │
│  src=0                                                                  │
│                                                                         │
│  SCATTER (one-to-each)                                                 │
│  ┌───────────────────────┐            ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│  │   A   B   C   D       │       →    │  A  │ │  B  │ │  C  │ │  D  │ │
│  └───────────────────────┘            └─────┘ └─────┘ └─────┘ └─────┘ │
│  src=0                                                                  │
│                                                                         │
│  GATHER (all-to-one)                                                   │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      ┌───────────────────────┐        │
│  │  A  │ │  B  │ │  C  │ │  D  │  →   │   A   B   C   D       │ (dst) │
│  └─────┘ └─────┘ └─────┘ └─────┘      └───────────────────────┘        │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

## Usage in Distributed Training

| Operation | Used By | Purpose |
|-----------|---------|---------|
| All-Reduce | DDP | Synchronize gradients across all GPUs |
| All-Gather | FSDP, Pipeline | Gather model shards for forward/backward |
| Reduce-Scatter | FSDP, ZeRO | Distribute reduced gradients to shards |
| Broadcast | Model init | Distribute initial weights from rank 0 |
| Send/Recv | Pipeline | Point-to-point activation transfer |

## Bandwidth vs Latency

```
Bandwidth-bound (large tensors):
┌────────────────────────────────────────┐
│ All-Reduce: 2 * (N-1)/N * data_size    │
│ All-Gather: (N-1)/N * data_size        │
│ Reduce-Scatter: (N-1)/N * data_size    │
└────────────────────────────────────────┘

Latency-bound (small tensors):
┌────────────────────────────────────────┐
│ Ring: 2 * (N-1) * latency              │
│ Tree: 2 * log(N) * latency             │
└────────────────────────────────────────┘
```

## Module Contents

### Scripts
- `collective_ops.py` - Interactive demo of all collective operations
- `bandwidth_test.py` - Measure actual NCCL bandwidth

### Learning Objectives

1. **Understand** each collective operation semantically
2. **Measure** communication bandwidth in your cluster
3. **Debug** NCCL issues with proper logging
4. **Optimize** by choosing appropriate collective sizes

## Quick Start

```bash
# Run collective operations demo
torchrun --nproc_per_node=4 collective_ops.py

# Run bandwidth test
torchrun --nproc_per_node=4 bandwidth_test.py --warmup 5 --iterations 20

# Enable NCCL debugging
NCCL_DEBUG=INFO torchrun --nproc_per_node=4 collective_ops.py
```

## NCCL Environment Variables

### Debugging
```bash
# Debug levels: WARN, INFO, TRACE
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL  # Or: INIT,COLL,P2P,NET

# Log to file
export NCCL_DEBUG_FILE=/tmp/nccl_debug.%h.%p.log
```

### Performance Tuning
```bash
# Buffer sizes (bytes)
export NCCL_BUFFSIZE=4194304  # 4MB default

# Number of channels
export NCCL_NCHANNELS_PER_NET_PEER=4

# Algorithms
export NCCL_ALGO=Ring  # Ring, Tree, CollNet

# Protocols  
export NCCL_PROTO=Simple  # Simple, LL, LL128
```

### Network Configuration
```bash
# Select network interface
export NCCL_SOCKET_IFNAME=eth0

# IB/RoCE settings
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0

# Disable P2P for debugging
export NCCL_P2P_DISABLE=0
```

## Common Issues and Solutions

### Issue: NCCL timeout
```bash
# Increase timeout
export NCCL_TIMEOUT=1800  # 30 minutes

# Or in code
torch.distributed.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(seconds=1800)
)
```

### Issue: OOM during collective
```bash
# Reduce buffer size
export NCCL_BUFFSIZE=2097152  # 2MB

# Or use gradient bucketing in DDP
model = DDP(model, bucket_cap_mb=25)
```

### Issue: Slow inter-node communication
```bash
# Check network interface
export NCCL_SOCKET_IFNAME=eth0

# Enable IB if available
export NCCL_IB_DISABLE=0
```

## Performance Guidelines

### Bucket Size Selection

| Tensor Size | Recommendation |
|-------------|----------------|
| < 1KB | Coalesce into larger buckets |
| 1KB - 10MB | Default settings work well |
| > 10MB | Consider gradient compression |

### Ring vs Tree Algorithm

| Scenario | Best Algorithm |
|----------|----------------|
| Few large tensors | Ring |
| Many small tensors | Tree |
| Hierarchical topology | Hybrid |

## Resources

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [PyTorch Distributed](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
