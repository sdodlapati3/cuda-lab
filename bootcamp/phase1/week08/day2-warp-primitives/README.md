# Day 2: Warp-Level Primitives

## Learning Objectives

- Master warp shuffle operations (__shfl_*)
- Use warp voting functions (__ballot, __any, __all)
- Understand warp synchronization

## Key Concepts

### What is a Warp?

- 32 threads executing in lockstep (SIMT)
- Same instruction, different data
- Implicit synchronization within warp

### Warp Shuffle Operations

Direct register-to-register communication:

| Function | Description |
|----------|-------------|
| `__shfl_sync(mask, val, src)` | Get value from lane `src` |
| `__shfl_up_sync(mask, val, delta)` | Get from lane `tid - delta` |
| `__shfl_down_sync(mask, val, delta)` | Get from lane `tid + delta` |
| `__shfl_xor_sync(mask, val, laneMask)` | Get from lane `tid ^ laneMask` |

### The Sync Mask

Always use `0xFFFFFFFF` for full warp:
```cuda
float val = __shfl_down_sync(0xFFFFFFFF, myval, 1);
```

### Warp Voting Functions

| Function | Returns |
|----------|---------|
| `__ballot_sync(mask, pred)` | Bitmask of threads where pred is true |
| `__any_sync(mask, pred)` | 1 if any thread has pred true |
| `__all_sync(mask, pred)` | 1 if all threads have pred true |
| `__popc(mask)` | Count of set bits |

### Why Use Warp Primitives?

1. **Faster** - No shared memory round-trip
2. **Simpler** - No explicit sync needed
3. **Less memory** - No shared memory allocation

## Exercises

1. **Warp reduce**: Sum using __shfl_down
2. **Broadcast**: Share one value to all
3. **Warp scan**: Prefix sum within warp

## Build & Run

```bash
./build.sh
./build/shuffle_demo
./build/voting_demo
```
