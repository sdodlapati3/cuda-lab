# Day 4: Bank Conflicts

## Learning Objectives

- Understand shared memory banks
- Identify bank conflicts
- Apply techniques to avoid conflicts

## Key Concepts

### Shared Memory Banks

Shared memory is divided into **32 banks** (one per warp lane).
- Each bank can serve one address per cycle
- Consecutive 4-byte words go to consecutive banks
- Address `i` → Bank `(i / 4) % 32`

```
Address:  0   4   8  12  16 ... 124  128 132 ...
Bank:     0   1   2   3   4 ...  31    0   1 ...
```

### Bank Conflicts

When multiple threads in a warp access **different addresses** in the **same bank**:

```cuda
// CONFLICT: All threads access bank 0!
smem[threadIdx.x * 32];  // Thread i accesses word i*32 → all bank 0

// NO CONFLICT: Each thread accesses different bank
smem[threadIdx.x];       // Thread i accesses bank i
```

### Conflict Types

| Scenario | Threads per Bank | Penalty |
|----------|------------------|---------|
| No conflict | 1 | 1 cycle |
| 2-way conflict | 2 | 2 cycles |
| 32-way conflict | 32 | 32 cycles |
| Broadcast | N (same address) | 1 cycle (special!) |

### Common Conflict Patterns

**Matrix column access**:
```cuda
// BAD: Column access with stride = row size (often 32)
smem[threadIdx.x][col];  // If columns are 32 wide → 32-way conflict!

// FIX: Pad rows
__shared__ float smem[32][33];  // 33 instead of 32
```

**Power-of-2 strides**:
```cuda
// BAD: Stride 8
smem[threadIdx.x * 8];  // 4 threads hit each bank

// FIX: Change algorithm or use padding
```

### Padding Technique

```cuda
// Without padding (32-way conflict on column access)
__shared__ float matrix[32][32];

// With padding (+1 element per row)
__shared__ float matrix[32][33];  // No conflict!
```

## Exercises

1. **Detect conflicts**: Profile with Nsight Compute
2. **Fix matrix access**: Apply padding
3. **Transpose optimization**: Conflict-free matrix transpose

## Build & Run

```bash
./build.sh
./build/bank_conflicts
./build/transpose
```
