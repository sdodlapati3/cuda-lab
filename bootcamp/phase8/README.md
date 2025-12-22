# Phase 8: ML Stack & Multi-GPU (Weeks 41-48)

Ship kernels as usable components and scale to multi-GPU systems.

## Overview

| Week | Topic | Focus |
|------|-------|-------|
| 41 | PyTorch Extensions Basics | setup.py, tensor accessors, build system |
| 42 | Autograd Integration | forward/backward, gradient checking |
| 43 | Triton Programming | Block-level GPU programming in Python |
| 44 | torch.compile & Inductor | How PyTorch generates kernels |
| 45 | NCCL Fundamentals | Collective operations, ring/tree algorithms |
| 46 | NCCL Advanced | Debugging, optimization, custom collectives |
| 47 | Multi-GPU Patterns | Overlap, streams, pipeline parallelism |
| 48 | Distributed Training | Gradient bucketing, scaling, phase summary |

## Directory Structure

```
phase8/
├── week41/          # PyTorch Extensions Basics (6 days)
├── week42/          # Autograd Integration (6 days)
├── week43/          # Triton Programming (6 days)
├── week44/          # torch.compile (6 days)
├── week45/          # NCCL Fundamentals (6 days)
├── week46/          # NCCL Advanced (6 days)
├── week47/          # Multi-GPU Patterns (6 days)
└── week48/          # Distributed Training (6 days)
```

## Key Technologies

- PyTorch C++/CUDA Extensions
- Triton DSL
- torch.compile / TorchInductor
- NCCL collective operations
- Multi-GPU synchronization

## Learning Objectives
- Build production PyTorch CUDA extensions
- Write Triton kernels and compare to CUDA
- Understand torch.compile internals
- Master NCCL collective operations
- Scale to multi-GPU systems

## Prerequisites
- Phase 1-7 completion
- PyTorch familiarity
- Basic distributed systems concepts
