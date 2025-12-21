# Day 4: Bank Conflicts in Shared Memory

## Learning Objectives
- Understand shared memory bank organization
- Identify and diagnose bank conflicts
- Apply padding to eliminate conflicts
- Measure performance impact

## Shared Memory Banks

### Bank Organization (A100)
- 32 banks, 4 bytes (1 float) per bank
- Consecutive 4-byte words → consecutive banks
- Bank = address % 32

### Bank Conflict
When multiple threads in a warp access different addresses in the same bank:
- 2-way conflict: 2× slowdown
- N-way conflict: N× slowdown

### No Conflict Cases
1. All threads access same address (broadcast)
2. Each thread accesses different bank

## GEMM Bank Conflict Analysis

### Shared Memory Layout
```cpp
__shared__ float As[TILE][TILE];  // [row][col]
```

### A Tile Access (during compute)
```cpp
As[threadIdx.y][k]  // Reading row
```
- Threads in same row (same threadIdx.y) read same element → OK (broadcast)
- But different iterations may cause conflicts

### B Tile Access (during compute)
```cpp
Bs[k][threadIdx.x]  // Reading column
```
- When TILE = 32 (matches bank count):
  - Threads 0, 32 access bank 0 → conflict!
  - Actually, column access patterns can cause conflicts

## Solution: Padding

```cpp
__shared__ float As[TILE][TILE + 1];  // +1 padding
__shared__ float Bs[TILE][TILE + 1];
```

Adding 1 column shifts bank assignment:
- Row 0, Col 0 → Bank 0
- Row 1, Col 0 → Bank 1 (not 0!)
- No more column access conflicts

## Exercises
1. Profile kernel with and without padding
2. Count bank conflicts using Nsight Compute
3. Visualize bank assignment
4. Measure speedup from padding
