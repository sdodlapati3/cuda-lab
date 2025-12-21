# Week 5: Execution Model

> **Goal:** Deeply understand how GPU threads are organized and executed.

## Daily Schedule

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 1 | Thread Hierarchy | Threads, warps, blocks, grids |
| 2 | Indexing Patterns | 1D, 2D, 3D indexing |
| 3 | Grid-Stride Loops | Handling arbitrary sizes |
| 4 | Warp Execution | SIMT and warp divergence |
| 5 | Launch Configuration | Choosing block and grid sizes |
| 6 | Streams & Events | Async execution basics |

## Prerequisites

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi
```

## Quick Start

```bash
cd day1-thread-hierarchy
./build.sh
./build/thread_demo
```

## Key Mental Models

### Thread Hierarchy
```
Grid (kernel launch)
├── Block (0,0) - runs on one SM
│   ├── Warp 0: threads 0-31 (execute in lockstep)
│   ├── Warp 1: threads 32-63
│   └── ...up to 1024 threads per block
├── Block (0,1)
└── ...potentially millions of blocks
```

### Hardware Mapping
| Software | Hardware |
|----------|----------|
| Thread | CUDA core execution |
| Warp (32 threads) | Scheduling unit |
| Block | Assigned to one SM |
| Grid | Entire GPU |

## Week 5 Checklist

- [ ] Can calculate global thread index in 1D, 2D, 3D
- [ ] Understand warp execution and divergence
- [ ] Can write grid-stride loops for any size
- [ ] Know launch configuration constraints
- [ ] Understand streams and async execution

## Official Doc References

| Topic | Programming Guide Section |
|-------|---------------------------|
| Thread Hierarchy | Ch. 2.2 |
| Built-in Variables | Ch. 4.1 |
| Kernel Launch | Ch. 2.1 |
| Streams | Ch. 3.2.7 |
