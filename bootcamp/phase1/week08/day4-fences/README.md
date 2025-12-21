# Day 4: Memory Fences

## Learning Objectives

- Understand memory ordering and visibility
- Use __threadfence() functions correctly
- Know when fences are required

## Key Concepts

### Memory Visibility Problem

Without fences, writes may not be visible to other threads:
```cuda
// Thread A                     // Thread B
data[idx] = value;              // May see old value!
flag = 1;                       // Even if flag is 1
```

### Memory Fence Functions

| Function | Scope | Use Case |
|----------|-------|----------|
| `__threadfence_block()` | Block | Visibility within block |
| `__threadfence()` | Device | Visibility to all GPU threads |
| `__threadfence_system()` | System | Visibility to CPU and other GPUs |

### What Fences Do

1. **Order writes** - All writes before fence complete before fence
2. **Visibility** - Writes become visible to other threads
3. **NOT a barrier** - Other threads don't wait

### Common Pattern: Producer-Consumer

```cuda
// Producer
data = value;
__threadfence();  // Ensure data is visible
flag = 1;         // Signal ready

// Consumer
while (flag == 0);  // Wait for signal
__threadfence();    // Ensure we see data
use(data);          // Now safe
```

### Atomics and Fences

Atomics have built-in memory ordering:
- atomicAdd, etc. include acquire/release semantics
- Often no explicit fence needed with atomics

## Exercises

1. **Visibility demo**: See the problem without fences
2. **Producer-consumer**: Implement with fences
3. **Block-level reduction**: When is fence needed?

## Build & Run

```bash
./build.sh
./build/fence_demo
```
