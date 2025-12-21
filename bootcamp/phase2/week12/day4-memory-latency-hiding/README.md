# Day 4: Memory Latency Hiding

## Learning Objectives

- Learn prefetching techniques
- Understand async memory operations
- Master double buffering

## Key Concepts

### Why Memory Latency Hurts

```
Without hiding:
  Load → WAIT 400 cycles → Compute → Store
  
With hiding:
  Load A → Load B → Load C → Compute A → Compute B → Compute C
  (Loads overlap with compute!)
```

### Double Buffering Pattern

```
Iteration 0:  [Load buf0] [          ]
Iteration 1:  [Load buf1] [Proc buf0 ]
Iteration 2:  [Load buf0] [Proc buf1 ]
Iteration 3:  [Load buf1] [Proc buf0 ]
...
```

### Async Copy (Ampere+)

```cpp
// Old way - synchronous
smem[tid] = gmem[idx];

// New way - async (hardware accelerated)
__pipeline_memcpy_async(&smem[tid], &gmem[idx], sizeof(float));
__pipeline_commit();
// ... do other work ...
__pipeline_wait_prior(0);
```

## Build & Run

```bash
./build.sh
./build/memory_latency
```
