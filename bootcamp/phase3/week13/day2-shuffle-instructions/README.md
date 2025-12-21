# Day 2: Shuffle Instructions

## Learning Objectives

- Master all four shuffle variants
- Understand when to use each
- Replace shared memory with shuffles

## Key Concepts

### The Four Shuffle Functions

```cpp
// All require a mask for active threads (usually 0xffffffff for full warp)

// Get value from specific lane
T __shfl_sync(mask, T var, int srcLane);

// Get value from lane - delta (for scans)
T __shfl_up_sync(mask, T var, int delta);

// Get value from lane + delta (for reductions)
T __shfl_down_sync(mask, T var, int delta);

// Get value from lane XOR mask (butterfly pattern)
T __shfl_xor_sync(mask, T var, int laneMask);
```

### Visual: Shuffle Operations

```
__shfl_sync(0xffffffff, val, 3):
  ALL lanes get value from lane 3
  
__shfl_down_sync(0xffffffff, val, 2):
  Lane 0 ← Lane 2
  Lane 1 ← Lane 3
  Lane 2 ← Lane 4
  ...
  Lane 29 ← Lane 31
  Lane 30, 31 ← unchanged
  
__shfl_up_sync(0xffffffff, val, 2):
  Lane 0, 1 ← unchanged
  Lane 2 ← Lane 0
  Lane 3 ← Lane 1
  ...
  Lane 31 ← Lane 29

__shfl_xor_sync(0xffffffff, val, 1):
  Lane 0 ↔ Lane 1
  Lane 2 ↔ Lane 3
  ...
  (pairwise exchange)
```

### Why Shuffles Are Fast

| Method | Latency | Sync Required? |
|--------|---------|----------------|
| Shared Memory | ~30 cycles | Yes (__syncthreads) |
| Shuffle | ~2 cycles | No (implicit in warp) |

## Build & Run

```bash
./build.sh
./build/shuffle_demo
```
