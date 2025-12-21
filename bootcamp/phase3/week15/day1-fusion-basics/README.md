# Day 1: Kernel Fusion Basics

## Learning Objectives

- Understand why kernel fusion matters
- Identify fusion opportunities
- Implement simple fused kernels

## Key Concepts

### Why Fuse Kernels?

1. **Reduce kernel launch overhead** (~5-10μs per launch)
2. **Eliminate intermediate memory traffic**
3. **Keep data in registers/shared memory**

### The Problem with Separate Kernels

```cpp
// Bad: 3 kernel launches, 6 memory transactions
__global__ void add(float* a, float* b, float* c, int n);
__global__ void mul(float* c, float s, float* d, int n);
__global__ void sqrt_k(float* d, float* e, int n);

add<<<...>>>(a, b, c, n);  // Read a,b; Write c
mul<<<...>>>(c, 2.0f, d, n);  // Read c; Write d
sqrt_k<<<...>>>(d, e, n);  // Read d; Write e
```

### The Fused Solution

```cpp
// Good: 1 kernel, 2 memory transactions
__global__ void fused(float* a, float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float c = a[idx] + b[idx];  // Register
        float d = c * 2.0f;          // Register
        out[idx] = sqrtf(d);         // Only write
    }
}
```

### Fusion Criteria

1. **Same iteration space** - Operations on same indices
2. **No dependencies between threads** - Each output independent
3. **Register pressure acceptable** - Don't spill to local memory

## Build & Run

```bash
./build.sh
./build/fusion_basics
```
