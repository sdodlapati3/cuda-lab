# Week 18: MIG & Production CUDA

## Overview

This week covers **Multi-Instance GPU (MIG)** technology and **production-grade CUDA practices**. MIG is available on NVIDIA A100, H100, and newer architectures.

## Prerequisites

- Completed Weeks 1-17
- Access to MIG-capable GPUs (A100/H100)
- Understanding of GPU memory hierarchy

## Learning Objectives

By the end of this week, you will:
1. Understand MIG architecture and partitioning
2. Configure MIG instances on H100/A100
3. Implement robust error handling and logging
4. Design production-ready CUDA applications

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 1 | MIG Fundamentals | GPU instances, memory partitions |
| 2 | MIG Configuration | nvidia-smi mig, compute instances |
| 3 | Error Management | Robust error handling, logging |
| 4 | Production Patterns | Deployment, monitoring, best practices |

## HPC Cluster Usage

MIG requires specific GPU models. On the ODU cluster:

```bash
# Request H100 node (supports MIG)
srun --partition=h100flex --gres=gpu:1 --time=01:00:00 --pty bash

# Request A100 node (supports MIG)
srun --partition=a100flex --gres=gpu:1 --time=01:00:00 --pty bash

# Note: MIG configuration typically requires admin privileges
# These exercises focus on querying MIG state and running on MIG instances
```

## Key APIs

```cpp
// Device Properties
cudaDevAttrMigMode                      // Check if MIG is enabled
cudaDeviceGetAttribute()                // Query MIG capabilities

// Environment Variables
CUDA_VISIBLE_DEVICES="MIG-..."         // Target specific MIG instance
CUDA_MPS_PIPE_DIRECTORY                // For MIG + MPS

// nvidia-smi Commands (admin)
nvidia-smi mig -lgip                   // List GPU instance profiles
nvidia-smi mig -lci                    // List compute instances
```

## Resources

- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [CUDA Error Handling Best Practices](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Production Deployment Guide](https://developer.nvidia.com/blog/cuda-pro-tip-always-set-the-current-device/)
