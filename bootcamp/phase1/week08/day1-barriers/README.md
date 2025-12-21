# Day 1: Barriers & __syncthreads()

## Learning Objectives

- Understand block-level synchronization
- Know when __syncthreads() is required
- Avoid deadlock from divergent sync

## Key Concepts

### What is __syncthreads()?

A barrier that blocks all threads in a block until every thread reaches it.

```cuda
sdata[tid] = input[idx];
__syncthreads();  // ALL threads must reach here
// Now safe to read any sdata[*]
float neighbor = sdata[tid + 1];
```

### When is it Required?

1. **After writing to shared memory** before other threads read
2. **Before reading shared memory** written by other threads
3. **At each reduction step** (ping-pong pattern)

### The Deadlock Danger

**NEVER** put __syncthreads() in divergent code!

```cuda
// DEADLOCK - some threads never reach sync!
if (tid < 32) {
    __syncthreads();  // Threads 32+ never get here
}
```

### Correct Pattern

```cuda
// ALL threads execute same sync
if (tid < 32) {
    // Work for first warp
}
__syncthreads();  // Everyone syncs here
```

### Cost of __syncthreads()

- Not free: ~few cycles + pipeline stall
- But necessary for correctness
- Minimize by restructuring algorithm

## Exercises

1. **Missing sync bug**: Find the race condition
2. **Proper sync patterns**: Fix the bugs
3. **Sync optimization**: Reduce unnecessary syncs

## Build & Run

```bash
./build.sh
./build/sync_demo
./build/sync_patterns
```
