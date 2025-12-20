# Exercise: Bank Conflicts

## Objective
Understand shared memory bank conflicts and learn techniques to avoid them.

## Background

Shared memory is divided into 32 banks (4 bytes each). When multiple threads in a warp access different addresses in the same bank, accesses serialize (bank conflict).

### No Conflict
- Each thread accesses a different bank
- All 32 accesses happen in parallel

### Bank Conflict
- Multiple threads access same bank (different addresses)
- Accesses serialize (2-way, 4-way, etc.)

### Bank Index Formula
```
bank_index = (address / 4) % 32
```

## Task

1. Implement matrix transpose with bank conflicts (naive)
2. Implement matrix transpose without bank conflicts (padded)
3. Measure the performance difference

## Key Insight

Adding padding to shared memory arrays can eliminate bank conflicts:
```cpp
// With conflicts (32 threads access bank 0)
__shared__ float tile[32][32];

// Without conflicts (padding shifts columns)
__shared__ float tile[32][33];  // +1 padding
```

## Files

- `bank_conflicts.cu` - Skeleton
- `solution.cu` - Reference
- `Makefile` - Build
- `test.sh` - Test
