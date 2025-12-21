# Day 1: Warp Fundamentals

## Learning Objectives

- Understand the SIMT execution model
- Learn how warps are formed and scheduled
- Recognize and avoid warp divergence

## Key Concepts

### What is a Warp?

A warp is a group of 32 threads that execute in lockstep:
- All 32 threads execute the **same instruction** at the **same time**
- This is called SIMT: Single Instruction, Multiple Threads
- The warp is the true unit of scheduling on the SM

### Warp Formation

```
Block of 128 threads:
┌─────────────────────────────────────────────────────────┐
│ Warp 0: threads 0-31   │ Warp 1: threads 32-63         │
│ Warp 2: threads 64-95  │ Warp 3: threads 96-127        │
└─────────────────────────────────────────────────────────┘

Threads are assigned to warps by threadIdx in row-major order.
```

### Warp Divergence

When threads in a warp take different paths:

```cpp
// DIVERGENCE: threads take different branches
if (threadIdx.x % 2 == 0) {
    do_even();  // Half the warp does this
} else {
    do_odd();   // Other half does this
}
// Both paths execute SERIALLY!

// NO DIVERGENCE: all threads same path
if (threadIdx.x < 32) {  // Entire warp 0
    do_work();
}
```

### Lane ID

Within a warp, each thread has a "lane" (0-31):
```cpp
int lane = threadIdx.x % 32;  // Or use: threadIdx.x & 31
int warp_id = threadIdx.x / 32;
```

## Build & Run

```bash
./build.sh
./build/warp_basics
```
