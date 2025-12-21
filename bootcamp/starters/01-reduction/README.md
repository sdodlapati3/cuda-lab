# Starter 01: Warp-Shuffle Reduction

**The Foundation Pattern** - Master this, and you understand 80% of CUDA optimization.

## Why This Matters

Reduction is the "Hello World" of parallel algorithms, but it teaches concepts that appear EVERYWHERE:
- Warp shuffle → Used in softmax, layernorm, attention
- Block-level coordination → Used in GEMM, scan, histogram
- Multiple elements per thread → Universal optimization

## Files

| File | Purpose |
|------|---------|
| `reduction_warp.cu` | Complete implementation with 3 versions |
| `Makefile` | Build and run |

## Build & Run

```bash
make
./reduction

# With custom size
./reduction 33554432  # 32M elements
```

## Expected Output

```
╔════════════════════════════════════════════════════════════════╗
║         WARP-SHUFFLE REDUCTION BENCHMARK                       ║
╠════════════════════════════════════════════════════════════════╣
║ Device: NVIDIA A100-SXM4-80GB                                  ║
║ Peak Bandwidth: 2039.0 GB/s                                    ║
║ Elements: 16777216 (64 MB)                                     ║
╠════════════════════════════════════════════════════════════════╣
V1: Naive (baseline)     :   234.56 μs |   273.1 GB/s |  13.4% peak
V2: Warp shuffle         :    98.23 μs |   652.4 GB/s |  32.0% peak
V3: Multi-element (8x)   :    42.15 μs |  1520.3 GB/s |  74.6% peak
╠════════════════════════════════════════════════════════════════╣
║ Verification: Expected=8388608.00, Got=8388607.50, Diff=0.00%  ║
╚════════════════════════════════════════════════════════════════╝
```

## The Three Versions

### V1: Naive (Baseline)
```
Every level: shared memory + __syncthreads()
            ↓
   [Thread sync overhead dominates]
```

### V2: Warp Shuffle
```
Within warp: __shfl_down_sync (no sync needed!)
Between warps: shared memory (one sync)
            ↓
   [Eliminated 5 of 8 syncs]
```

### V3: Multiple Elements Per Thread
```
Each thread loads 8 elements → local sum
Then warp reduction
            ↓
   [8x fewer reduction operations]
```

## Key Code Patterns

### Warp Reduction (Memorize This!)
```cuda
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // Only lane 0 has final result
}
```

### Multiple Elements Per Thread
```cuda
float thread_sum = 0.0f;
#pragma unroll
for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
    int idx = start + i * BLOCK_SIZE;  // Strided for coalescing!
    thread_sum += input[idx];
}
```

## Exercises

1. **Add V4:** Use vectorized loads (`float4`) for even better bandwidth
2. **Implement `warp_reduce_max()`:** Replace `+` with `fmaxf`
3. **Find-index-of-max kernel:** Track both value and index
4. **Profile with Nsight Compute:** Identify remaining bottlenecks
5. **Two-pass reduction:** For arrays larger than can fit in one kernel

## What You Learn Here Applies To

| Pattern | Used In |
|---------|---------|
| Warp shuffle | Softmax, LayerNorm, Attention |
| Block reduction | Histogram, Scan |
| Multi-element | Every memory-bound kernel |
| Grid-stride loop | Universal pattern |

## Next Steps

After mastering this:
1. → Implement scan (uses same warp primitives)
2. → Implement histogram (adds atomics)
3. → Study softmax (reduction + max + division)
