# Week 9: CUDA Streams & Concurrency

## Learning Philosophy

> **CUDA C++ First, Python/Numba as Optional Backup**

All notebooks show CUDA C++ code as the PRIMARY implementation. Python/Numba is provided optionally for quick interactive testing in Colab.

## Overview

Master concurrent execution and async operations for maximum GPU utilization.

## Daily Schedule

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Stream Basics | Default stream, stream creation, async operations |
| 2 | Overlapping Transfers | cudaMemcpyAsync, pinned memory, H2D/D2H overlap |
| 3 | Multi-Stream Execution | Concurrent kernels, stream priorities |
| 4 | Events & Synchronization | cudaEvent, inter-stream sync, timing |
| 5 | Practice & Quiz | Exercises + checkpoint assessment |

## Prerequisites
- Week 8: Profiling & Analysis
- Understanding of kernel launches
- Memory management basics

## Key Skills
- [ ] Create and manage CUDA streams
- [ ] Overlap data transfer with computation
- [ ] Run multiple kernels concurrently
- [ ] Use events for synchronization and timing
- [ ] Profile stream concurrency with Nsight Systems
