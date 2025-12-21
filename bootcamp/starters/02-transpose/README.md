# Starter 02: Tiled Transpose

**The Shared Memory Pattern** - This pattern is the building block of GEMM.

## Why This Matters

Transpose seems simple, but it teaches the CORE pattern used in:
- **GEMM:** Load tiles to shared memory → compute → store
- **Convolution:** Same pattern with different access
- **Any layout change:** Image format conversion, tensor reshaping

The bank conflict lesson is worth the entire exercise.

## Files

| File | Purpose |
|------|---------|
| `transpose_tiled.cu` | Complete implementation with 3 versions |
| `Makefile` | Build and run |

## Build & Run

```bash
make
./transpose

# With custom size
./transpose 8192 8192
```

## Expected Output

```
╔════════════════════════════════════════════════════════════════╗
║         TILED TRANSPOSE BENCHMARK                              ║
╠════════════════════════════════════════════════════════════════╣
║ Device: NVIDIA A100-SXM4-80GB                                  ║
║ Peak Bandwidth: 2039.0 GB/s                                    ║
║ Matrix: 4096 × 4096 (64 MB)                                    ║
╠════════════════════════════════════════════════════════════════╣
V1: Naive (uncoalesced)       :   487.23 μs |   274.5 GB/s |  13.5% peak
V2: Tiled (bank conflicts)    :   152.34 μs |   878.3 GB/s |  43.1% peak
V3: Tiled + padding           :    82.56 μs |  1620.4 GB/s |  79.5% peak
╠════════════════════════════════════════════════════════════════╣
║ Verification: PASSED ✓                                         ║
╚════════════════════════════════════════════════════════════════╝
```

## The Three Versions

### V1: Naive (Baseline)
```
Read:  input[y][x]  → coalesced ✓
Write: output[x][y] → NOT coalesced ✗ (stride = width)
```

### V2: Tiled (with Bank Conflicts)
```
1. Load tile to shared memory (coalesced)
2. __syncthreads()
3. Store transposed (coalesced globally, but...)

Shared memory access: tile[threadIdx.x][threadIdx.y]
  → 32 threads access column = 32-way bank conflict!
```

### V3: Tiled + Padding (No Bank Conflicts)
```
tile[TILE_DIM][TILE_DIM + 1]  ← The +1 is magic!

Without padding: Column accesses hit same bank
With padding:    Column accesses hit different banks
```

## Key Visualization

### Bank Conflict Problem
```
Shared memory banks (32 banks, 4 bytes each):

Bank:    0    1    2    3   ...   31
Row 0: [0,0][0,1][0,2][0,3] ... [0,31]
Row 1: [1,0][1,1][1,2][1,3] ... [1,31]
Row 2: [2,0][2,1][2,2][2,3] ... [2,31]
...

Reading column 0: [0,0], [1,0], [2,0]...
  → All in Bank 0 → 32-way conflict!
```

### Padding Solution
```
Bank:    0    1    2    3    4   ...
Row 0: [0,0][0,1][0,2]...[0,31][pad]
Row 1: [1,0][1,1][1,2]...[1,31][pad]
Row 2: [2,0][2,1][2,2]...[2,31][pad]

Reading column 0: 
  [0,0] → Bank 0
  [1,0] → Bank 1 (shifted by padding!)
  [2,0] → Bank 2
  → No conflicts!
```

## Critical Code Pattern

```cuda
// WITHOUT padding (bank conflicts)
__shared__ float tile[32][32];
tile[threadIdx.x][threadIdx.y];  // Column access = conflict

// WITH padding (no conflicts)
__shared__ float tile[32][33];   // +1 padding
tile[threadIdx.x][threadIdx.y];  // Column access = no conflict
```

## Exercises

1. **Profile with Nsight Compute:** Compare bank conflict metrics between V2 and V3
2. **TILE_DIM = 16:** Does padding still help? (Hint: think about bank width)
3. **Vectorized loads:** Use `float4` to load 4 elements at once
4. **Non-square matrices:** Fix the code for width ≠ height
5. **Swizzled access:** XOR-based indexing instead of padding

## What You Learn Here Applies To

| Concept | Used In |
|---------|---------|
| Tiled shared memory | GEMM, convolution |
| Bank conflict avoidance | Every shared memory kernel |
| Coalesced access patterns | Every global memory access |
| Padding trick | Common optimization |

## Next Steps

After mastering this:
1. → Implement tiled GEMM (same pattern, with computation)
2. → Study CUTLASS tile loading patterns
3. → Implement batched transpose for tensors
