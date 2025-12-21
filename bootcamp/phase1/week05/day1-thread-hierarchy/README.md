# Day 1: Thread Hierarchy

## Learning Objectives

- Understand threads, warps, blocks, and grids
- Know the hardware mapping
- Visualize thread organization

## Key Concepts

### The Hierarchy

```
Grid
├── gridDim.x × gridDim.y × gridDim.z blocks
│
Block
├── blockDim.x × blockDim.y × blockDim.z threads
├── Threads are grouped into warps (32 threads each)
│
Thread
├── Executes kernel code
├── Has its own registers
└── Identified by threadIdx.x, threadIdx.y, threadIdx.z
```

### Built-in Variables

| Variable | Type | Description |
|----------|------|-------------|
| `threadIdx` | dim3 | Thread index within block |
| `blockIdx` | dim3 | Block index within grid |
| `blockDim` | dim3 | Threads per block |
| `gridDim` | dim3 | Blocks per grid |
| `warpSize` | int | Threads per warp (always 32) |

### Global Thread Index (1D)

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

## Exercises

1. **Print Thread Info**: Each thread prints its indices
2. **Count Threads**: Verify total thread count
3. **Warp ID**: Calculate which warp each thread belongs to

## Build & Run

```bash
./build.sh
./build/thread_demo
```
