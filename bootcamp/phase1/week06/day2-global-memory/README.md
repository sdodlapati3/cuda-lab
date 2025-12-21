# Day 2: Global Memory & Coalescing

## Learning Objectives

- Understand memory coalescing
- Recognize coalesced vs non-coalesced access patterns
- Measure bandwidth impact of access patterns

## Key Concepts

### What is Coalescing?

When threads in a warp access **consecutive memory addresses**, the hardware combines them into fewer memory transactions.

**Coalesced (Good)**:
```
Thread 0 → data[0]
Thread 1 → data[1]
Thread 2 → data[2]
...
Thread 31 → data[31]
→ 1 memory transaction (128 bytes)
```

**Non-coalesced (Bad)**:
```
Thread 0 → data[0]
Thread 1 → data[32]    // Strided access
Thread 2 → data[64]
...
→ 32 memory transactions!
```

### Access Patterns

| Pattern | Coalesced | Efficiency |
|---------|-----------|------------|
| `data[idx]` | Yes | 100% |
| `data[idx * stride]` (stride > 1) | No | 1/stride |
| `data[random]` | No | Very low |
| `data[idx + offset]` | Yes | 100% (if aligned) |

### Memory Transaction Size

- Ampere: 32-byte, 64-byte, or 128-byte transactions
- Cache line: 128 bytes
- Ideal: 32 threads × 4 bytes = 128 bytes = 1 transaction

### AoS vs SoA

**Array of Structures (AoS)** - Bad for coalescing:
```cpp
struct Particle { float x, y, z, w; };
Particle particles[N];  // Thread i reads particles[i].x (stride of 16 bytes!)
```

**Structure of Arrays (SoA)** - Good for coalescing:
```cpp
struct Particles {
    float x[N], y[N], z[N], w[N];
};  // Thread i reads x[i] (consecutive!)
```

## Exercises

1. **Coalescing benchmark**: Compare stride-1 vs stride-N access
2. **AoS vs SoA**: Convert struct layout and measure speedup
3. **Profile with Nsight**: See memory efficiency metrics

## Build & Run

```bash
./build.sh
./build/coalescing_demo
./build/aos_vs_soa
```
