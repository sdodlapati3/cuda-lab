# Day 3: Shared Memory Basics

## Learning Objectives

- Understand shared memory purpose and usage
- Write kernels using shared memory
- Understand block-level synchronization

## Key Concepts

### What is Shared Memory?

- Fast on-chip memory (L1 speed: ~5 cycles)
- Shared by all threads in a block
- Programmer-managed scratchpad
- 48-164 KB per SM (configurable)

### Use Cases

1. **Reduce global memory traffic**: Load once, use many times
2. **Inter-thread communication**: Share data within block
3. **Tiling**: Work on data chunks that fit in shared memory
4. **Reduction/Scan**: Block-level parallel patterns

### Static vs Dynamic Shared Memory

**Static (compile-time size)**:
```cuda
__global__ void kernel() {
    __shared__ float smem[256];  // Fixed size
}
```

**Dynamic (runtime size)**:
```cuda
__global__ void kernel() {
    extern __shared__ float smem[];  // Size set at launch
}

// Launch with shared memory size
kernel<<<blocks, threads, shared_bytes>>>();
```

### Synchronization

**Critical**: Shared memory requires synchronization!

```cuda
__shared__ float smem[256];

// Thread 0 writes
if (threadIdx.x == 0) {
    smem[0] = 42.0f;
}

__syncthreads();  // BARRIER: All threads wait here

// Now all threads can safely read smem[0]
float val = smem[0];
```

### Common Pattern: Load-Compute-Store

```cuda
__shared__ float tile[TILE_SIZE];

// 1. Load from global to shared (coalesced)
tile[threadIdx.x] = global_data[idx];
__syncthreads();

// 2. Compute using shared memory (fast)
float result = compute(tile);
__syncthreads();

// 3. Store back to global
global_data[idx] = result;
```

## Exercises

1. **Reverse array**: Use shared memory to reverse within block
2. **Block sum**: Sum elements using shared memory
3. **Moving average**: Use shared memory for sliding window

## Build & Run

```bash
./build.sh
./build/shared_basics
./build/block_reduction
```
