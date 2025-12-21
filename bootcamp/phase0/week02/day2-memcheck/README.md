# Day 2: Memory Error Detection

## What You'll Learn

- Detect out-of-bounds memory access
- Find uninitialized memory reads
- Detect memory leaks
- Understand memcheck vs racecheck

## The Tool: compute-sanitizer --tool memcheck

```bash
compute-sanitizer --tool memcheck ./your_program
```

This detects:
- **Out-of-bounds access** (reading/writing past array end)
- **Misaligned access** (not aligned to data type size)
- **Invalid global memory access**
- **Memory leaks** (with `--leak-check full`)

## Quick Start

```bash
./build.sh
compute-sanitizer --tool memcheck ./memory_errors
compute-sanitizer --tool memcheck --leak-check full ./memory_leaks
```

## Common Memory Errors

### 1. Out-of-Bounds Access

```cpp
__global__ void oob_access(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // BUG: No bounds check!
    data[idx] = idx;  // Crashes if idx >= n
}

// FIX:
__global__ void bounds_checked(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx;
    }
}
```

### 2. Uninitialized Memory

```cpp
__global__ void uninit_read(int* output) {
    __shared__ int sdata[256];
    // BUG: sdata not initialized!
    output[threadIdx.x] = sdata[threadIdx.x];
}

// FIX:
__global__ void init_first(int* output) {
    __shared__ int sdata[256];
    sdata[threadIdx.x] = 0;  // Initialize!
    __syncthreads();
    output[threadIdx.x] = sdata[threadIdx.x];
}
```

### 3. Invalid Pointer

```cpp
__global__ void null_deref(int* data) {
    if (data == nullptr) return;  // Check added, but...
    // What if only some threads have null?
    *data = 42;
}

// This is architecture-dependent. Always check before launch!
```

### 4. Memory Leaks

```cpp
void leaky_function() {
    int* d_data;
    cudaMalloc(&d_data, 1024);
    // BUG: No cudaFree!
}  // Memory leaked

// FIX: Always free
void clean_function() {
    int* d_data;
    cudaMalloc(&d_data, 1024);
    // ... use it ...
    cudaFree(d_data);  // Clean up!
}
```

## memcheck Output

```
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at 0x00000148 in kernel(int*, int)
=========     by thread (256,0,0) in block (0,0,0)
=========     Address 0x7f1234567400 is out of bounds
=========
========= ERROR SUMMARY: 1 error
```

## Leak Detection

```bash
compute-sanitizer --tool memcheck --leak-check full ./your_program
```

Output:
```
========= Leaked 1024 bytes at 0x7f1234560000
=========     Saved host backtrace up to driver entry point
=========     Host Frame: cudaMalloc
=========     Host Frame: leaky_function at memory_leaks.cu:42
```

## initcheck Tool

For uninitialized device memory:
```bash
compute-sanitizer --tool initcheck ./your_program
```

## Exercises

1. Run `compute-sanitizer --tool memcheck ./memory_errors`
2. Identify each type of error
3. Fix the errors in your copy of the code
4. Verify with `--leak-check full`
5. Try `--tool initcheck` for uninitialized reads

## Debug Build Requirement

For accurate line numbers, build with `-G -lineinfo`:

```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")
```

**Warning:** `-G` disables optimizations. Only use for debugging!
