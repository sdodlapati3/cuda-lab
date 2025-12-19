# Week 2: Memory Patterns & Optimization

Welcome to Week 2! This week focuses on **memory optimization** - the most critical skill for high-performance CUDA code.

## 📋 Overview

This week you'll master:
- Global memory access patterns
- Memory coalescing for maximum bandwidth
- Shared memory for thread cooperation
- Bank conflicts and how to avoid them
- Constant and texture memory

## 📅 Daily Schedule

| Day | Topic | Time | Materials |
|-----|-------|------|-----------|
| **Day 1** | Memory Coalescing | 4-5 hrs | [day-1-memory-coalescing.ipynb](day-1-memory-coalescing.ipynb) |
| **Day 2** | Shared Memory | 4-5 hrs | [day-2-shared-memory.ipynb](day-2-shared-memory.ipynb) |
| **Day 3** | Bank Conflicts | 4-5 hrs | [day-3-bank-conflicts.ipynb](day-3-bank-conflicts.ipynb) |
| **Day 4** | Special Memory Types | 3-4 hrs | [day-4-special-memory.ipynb](day-4-special-memory.ipynb) |
| **Day 5** | Project & Quiz | 4-5 hrs | Image filter project + [checkpoint-quiz.md](checkpoint-quiz.md) |

## ✅ Checklist

### By end of Day 1:
- [ ] Understand coalesced vs non-coalesced access
- [ ] Can identify access patterns in code
- [ ] Know the 32/64/128 byte transaction sizes

### By end of Day 2:
- [ ] Can declare and use shared memory
- [ ] Understand `__syncthreads()` usage
- [ ] Can implement tile-based algorithms

### By end of Day 3:
- [ ] Understand bank conflict causes
- [ ] Can identify conflict patterns
- [ ] Know padding and access pattern fixes

### By end of Day 4:
- [ ] Know when to use constant memory
- [ ] Understand texture memory benefits
- [ ] Can choose the right memory type

### By end of Day 5:
- [ ] Completed Gaussian blur project
- [ ] Quiz score ≥ 25/30
- [ ] Ready for Week 3!

## 🎯 Week Project: Image Filter

Implement a Gaussian blur using shared memory tiling:
- Load image tile into shared memory
- Apply convolution kernel
- Handle boundary conditions
- Compare performance vs naive implementation

## 🔗 Quick Links

- [CUDA Memory Guide](../../cuda-programming-guide/02-basics/cuda-memory.md)
- [Device Memory Access](../../cuda-programming-guide/03-advanced/device-memory-access.md)
- [Quick Reference](../../notes/cuda-quick-reference.md)

## 📝 Key Concepts This Week

### Memory Bandwidth
```
Global Memory: 200-900 GB/s (high latency)
Shared Memory: ~10 TB/s (low latency)
Registers: Fastest (per-thread)
```

### The Coalescing Rule
```
Adjacent threads should access adjacent memory locations
Thread 0 → mem[0], Thread 1 → mem[1], Thread 2 → mem[2]...
```

### Shared Memory Tiling Pattern
```
1. Load tile from global → shared memory
2. __syncthreads()
3. Compute using shared memory
4. __syncthreads()
5. Store results to global memory
```

---

[← Week 1](../week-01/README.md) | [12-Week Curriculum](../12-week-curriculum.md) | [Week 3 →](../week-03/README.md)
