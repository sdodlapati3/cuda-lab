# Day 3: Atomic Operations

## Learning Objectives

- Master atomic operations for thread-safe updates
- Understand performance implications
- Learn when atomics are appropriate

## Key Concepts

### Why Atomics?

Without atomics, concurrent updates cause race conditions:
```cuda
// BAD: Race condition!
counter++;  // Read-modify-write is NOT atomic

// GOOD: Thread-safe
atomicAdd(&counter, 1);
```

### Available Atomics

| Function | Operation | Types |
|----------|-----------|-------|
| `atomicAdd` | `*addr += val` | int, uint, float, double |
| `atomicSub` | `*addr -= val` | int, uint |
| `atomicMin` | `*addr = min(*addr, val)` | int, uint, float |
| `atomicMax` | `*addr = max(*addr, val)` | int, uint, float |
| `atomicExch` | `*addr = val` | int, float |
| `atomicCAS` | Compare-and-swap | int, uint |
| `atomicAnd/Or/Xor` | Bitwise ops | int, uint |

### atomicCAS (Compare-And-Swap)

The fundamental building block:
```cuda
int atomicCAS(int* addr, int compare, int val);
// If *addr == compare, set *addr = val
// Returns OLD value of *addr
```

Can implement ANY atomic operation!

### Performance Considerations

1. **Contention** - Many threads hitting same address = slow
2. **Location** - Shared memory atomics > Global memory atomics
3. **Hardware** - Newer GPUs have faster atomics

### Optimization Strategies

- Reduce contention with privatization
- Use warp aggregation before atomics
- Consider tree reduction instead of atomics

## Exercises

1. **Basic atomics**: Counter, min, max
2. **Custom atomic**: Implement atomicMul using atomicCAS
3. **Contention test**: Measure atomic performance

## Build & Run

```bash
./build.sh
./build/atomic_demo
./build/atomic_patterns
```
