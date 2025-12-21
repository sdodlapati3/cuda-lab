# Day 1: compute-sanitizer - Race Detection

## What You'll Learn

- Detect data races between threads
- Understand race conditions in GPU kernels
- Fix synchronization issues

## What is a Race Condition?

A race condition occurs when:
1. Multiple threads access the same memory location
2. At least one access is a write
3. There's no synchronization between them

On GPUs, this is **extremely common** due to thousands of parallel threads.

## The Tool: compute-sanitizer --tool racecheck

```bash
compute-sanitizer --tool racecheck ./your_program
```

This instruments your code to detect:
- **Write-after-write (WAW)** hazards
- **Read-after-write (RAW)** hazards  
- **Write-after-read (WAR)** hazards

## Quick Start

```bash
./build.sh
compute-sanitizer --tool racecheck ./race_example
```

## Example Race Condition

```cpp
// BAD: Race condition!
__global__ void bad_reduction(int* data, int* result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            *result += data[i];  // Race: all threads might do this!
        }
    }
}

// Actually, the race is subtle here. More obvious:
__global__ void obvious_race(int* shared_counter) {
    // All threads increment the same location!
    *shared_counter += 1;  // RACE!
}
```

## Fixing Race Conditions

### Option 1: Atomic Operations
```cpp
__global__ void fixed_with_atomics(int* counter) {
    atomicAdd(counter, 1);  // Thread-safe
}
```

### Option 2: Proper Synchronization
```cpp
__global__ void fixed_with_sync(int* data) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    
    sdata[tid] = data[tid];
    __syncthreads();  // All threads wait here
    
    // Now safe to read other threads' data
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += sdata[i];
        }
        data[0] = sum;
    }
}
```

### Option 3: Warp-Level Operations
```cpp
__global__ void fixed_with_warp(int* result) {
    int val = 1;
    // Warp-level reduction - no races within a warp
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(result, val);  // Still need atomic between warps
    }
}
```

## compute-sanitizer Output

```
========= COMPUTE-SANITIZER
========= Race reported between Write access at 0x00000148 in kernel
=========     and Read access at 0x00000150 in kernel
=========     Race detected on write to 0x7f1234567890
=========
========= ERROR SUMMARY: 1 error
```

## Common Race Patterns

| Pattern | Problem | Fix |
|---------|---------|-----|
| Global counter | All threads increment same location | atomicAdd |
| Shared memory reduction | Reading before all writes complete | __syncthreads |
| Warp divergent access | Different warp paths race | Careful sync |
| Block boundary | Threads in different blocks race | atomics or redesign |

## Exercises

1. Run `compute-sanitizer --tool racecheck ./race_example`
2. Identify which kernel has the race condition
3. Fix the race using atomics
4. Verify the fix with compute-sanitizer
5. Compare performance of atomic vs original (broken) version

## Performance Note

`compute-sanitizer` makes your code **10-100x slower**. Only use it for debugging, never in production.
