# Week 17: Inter-Process Communication & Textures

## Overview

This week covers two important advanced topics:
1. **IPC (Inter-Process Communication)** - Sharing GPU memory between processes
2. **Texture Memory** - Hardware-accelerated interpolation and caching

## Prerequisites
- Weeks 1-16 completed
- Access to multi-GPU system (HPC cluster recommended)
- Understanding of CUDA memory management

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | IPC Fundamentals | `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, memory sharing |
| 2 | Multi-Process Patterns | Producer-consumer, inference servers, NCCL patterns |
| 3 | Texture Objects | `cudaCreateTextureObject`, filtering, addressing modes |
| 4 | Texture Applications | Image processing, interpolation, lookup tables |

## Learning Objectives

After this week, you will be able to:
- Share GPU memory between separate processes
- Implement producer-consumer patterns with IPC
- Create and use texture objects for efficient data access
- Apply texture filtering for image processing

## HPC Cluster Usage

These notebooks work best on HPC clusters with multiple GPUs:

```bash
# Request an H100 node
srun --partition=h100flex --gres=gpu:1 --time=01:00:00 --pty bash

# For IPC examples (need 2 GPUs)
srun --partition=h100dualflex --gres=gpu:2 --time=01:00:00 --pty bash
```

## Files

- `day-1-ipc-fundamentals.ipynb` - IPC basics and memory handle sharing
- `day-2-multi-process-patterns.ipynb` - Advanced IPC patterns
- `day-3-texture-objects.ipynb` - Texture object creation and configuration
- `day-4-texture-applications.ipynb` - Practical texture applications
- `checkpoint-quiz.md` - Week 17 assessment
