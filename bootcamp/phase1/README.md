# Phase 1: CUDA Fundamentals

> **Duration:** 4 weeks (Weeks 5-8)
> **Goal:** Write correct kernels and deeply understand the execution model.

## Prerequisites

- ✅ Completed Phase 0 (build, debug, profile infrastructure)
- ✅ Read [daily-reference-spine.md](../phase0/daily-reference-spine.md)
- ✅ Familiar with [library-first-guide.md](../phase0/library-first-guide.md)

---

## Weekly Schedule

| Week | Topic | Focus |
|------|-------|-------|
| [Week 5](week05/) | Execution Model | Threads, warps, blocks, grids, indexing |
| [Week 6](week06/) | Memory Hierarchy | Global, shared, registers, coalescing |
| [Week 7](week07/) | First Real Kernels | Vector ops, SAXPY, reduction |
| [Week 8](week08/) | Synchronization | __syncthreads, atomics, transpose |

---

## Official Documentation Mapping

### CUDA Programming Guide Sections
| Topic | Section |
|-------|---------|
| Kernels | Ch. 2.1 |
| Thread Hierarchy | Ch. 2.2 |
| Memory Hierarchy | Ch. 2.3, Ch. 5 |
| Heterogeneous Programming | Ch. 2.4 |
| Compute Capability | Ch. 2.5 |

### CUDA Best Practices Guide Sections
| Topic | Section |
|-------|---------|
| Memory Optimizations | Ch. 5 |
| Execution Configuration | Ch. 10 |
| Instruction Optimization | Ch. 11 |

---

## Phase 1 Gate

**You can proceed to Phase 2 when:**

- [ ] Can explain the thread hierarchy (thread → warp → block → grid)
- [ ] Understand memory coalescing and can identify violations
- [ ] Know when to use shared memory vs registers
- [ ] Have written a reduction kernel achieving >70% memory bandwidth
- [ ] Can explain *why* a kernel is slow, not just that it is

---

## Deliverables

1. **Reduction kernel** at >70% of theoretical memory bandwidth
2. **Transpose kernel** with bank conflict analysis
3. **Written explanation** of coalescing patterns
4. **Profiler evidence** for all performance claims

---

## Key Mental Models

### The Thread Hierarchy
```
Grid
├── Block (0,0)
│   ├── Warp 0 (threads 0-31)
│   ├── Warp 1 (threads 32-63)
│   └── ...
├── Block (0,1)
│   └── ...
└── ...
```

### The Memory Hierarchy
```
Registers (fastest, per-thread)
    ↓
Shared Memory (fast, per-block)
    ↓
L1/L2 Cache (automatic)
    ↓
Global Memory (slow, per-device)
```

### The Execution Model
```
1. CPU launches kernel → GPU queues work
2. SM receives block → schedules warps
3. Warp executes 32 threads in lockstep
4. Memory access → coalesced or scattered
5. Synchronization → barriers and atomics
```

---

## Quick Start

```bash
# Week 5: Execution model
cd week05/day1-thread-hierarchy
./build.sh
./build/thread_demo

# Profile it
ncu --set full ./build/thread_demo
```

---

## Common Mistakes to Avoid

### 1. Index Out of Bounds
```cuda
// WRONG: No bounds check
__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = 1.0f;  // Crash if idx >= n
}

// RIGHT: Always check bounds
__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 1.0f;
    }
}
```

### 2. Race Conditions
```cuda
// WRONG: Race between threads
__global__ void kernel(int* counter) {
    (*counter)++;  // Multiple threads read-modify-write
}

// RIGHT: Use atomics
__global__ void kernel(int* counter) {
    atomicAdd(counter, 1);
}
```

### 3. Missing Synchronization
```cuda
// WRONG: Using shared data before all threads wrote
__shared__ float s[256];
s[threadIdx.x] = data[idx];
float val = s[threadIdx.x + 1];  // Other thread may not have written yet!

// RIGHT: Synchronize first
__shared__ float s[256];
s[threadIdx.x] = data[idx];
__syncthreads();  // Wait for all threads
float val = s[threadIdx.x + 1];
```

---

## Resources

- [CUDA Programming Guide - Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
- [CUDA Best Practices - Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [GPU Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)
