# Day 6: Lock-Free Patterns

## Learning Objectives

- Implement spinlocks on GPU
- Understand lock-free algorithm principles
- Learn progress guarantees

## Key Concepts

### Progress Guarantees

| Level | Guarantee |
|-------|-----------|
| Wait-free | ALL threads complete in finite steps |
| Lock-free | SOME thread completes in finite steps |
| Obstruction-free | Lone thread completes in finite steps |
| Blocking | Threads may wait forever |

GPU atomics typically provide lock-free guarantees.

### Spinlocks

Simple blocking primitive:
```cuda
__device__ void lock(int* mutex) {
    while (atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int* mutex) {
    atomicExch(mutex, 0);
}
```

### Lock-Free Patterns

**Compare-and-swap loop:**
```cuda
__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_int = (int*)addr;
    int old = *addr_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
```

**Lock-free stack (ABA-aware):**
```cuda
// Use tagged pointers or version counters
// to avoid ABA problem
```

### Why Lock-Free on GPU?

1. **No preemption**: Spinning thread blocks other warps
2. **Warp divergence**: Lock holders may stall
3. **Priority**: No thread priority = potential starvation
4. **Performance**: Atomics are already lock-free

### Guidelines

1. Prefer atomics over spinlocks
2. Keep critical sections tiny
3. Use lock-free algorithms when possible
4. Beware ABA problem

## Exercises

1. **Spinlock**: Implement and test
2. **Lock-free max**: Custom atomic operation
3. **Progress test**: Demonstrate starvation

## Build & Run

```bash
./build.sh
./build/lock_free_demo
./build/spinlock_demo
```
