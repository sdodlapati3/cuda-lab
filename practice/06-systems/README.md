# Practice Exercises: Systems Programming

This module contains practice exercises for advanced CUDA systems programming:

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-ipc-producer-consumer | IPC Memory Sharing | ⭐⭐⭐ |
| ex02-texture-image-processing | Texture Objects | ⭐⭐⭐ |
| ex03-error-handling | Production Error Handling | ⭐⭐⭐⭐ |
| ex04-gpu-monitor | GPU Health Monitoring | ⭐⭐⭐⭐ |

## Prerequisites

- Completed Weeks 17-18 of the learning path
- Understanding of CUDA memory model
- Experience with multi-process programming

## HPC Cluster Notes

These exercises benefit from HPC resources:

```bash
# For IPC exercises (need same-node processes)
srun --partition=h100flex --gres=gpu:1 --ntasks=2 --time=01:00:00 --pty bash

# For multi-GPU exercises
srun --partition=h100dualflex --gres=gpu:2 --time=01:00:00 --pty bash
```
