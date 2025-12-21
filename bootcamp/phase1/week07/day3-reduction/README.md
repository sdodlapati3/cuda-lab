# Day 3: Reduction

## Learning Objectives

- Implement parallel reduction (sum, min, max)
- Understand reduction complexity and patterns
- Achieve high memory bandwidth efficiency

## Key Concepts

### What is Reduction?

Combine N elements into 1 result using an associative operator:
- Sum: `result = a[0] + a[1] + ... + a[N-1]`
- Min: `result = min(a[0], a[1], ..., a[N-1])`
- Max: `result = max(a[0], a[1], ..., a[N-1])`
- Product: `result = a[0] * a[1] * ... * a[N-1]`

### Parallel Reduction Strategy

1. **First pass**: Each block reduces a portion to 1 element
2. **Second pass**: Reduce block results to final answer

### Evolution of Reduction Kernels

| Version | Problem | Bandwidth |
|---------|---------|-----------|
| Interleaved | Bank conflicts, divergence | ~10% |
| Sequential | Non-coalesced | ~30% |
| Sequential + unroll | Launch overhead | ~50% |
| Warp primitives | Shared memory | ~70% |
| CUB | Our implementation | ~90%+ |

### Performance Target

For A100:
- Peak bandwidth: ~2 TB/s
- Good reduction: >70% = >1.4 TB/s
- We read N elements = N * 4 bytes
- Target: (N * 4) / time > 1.4 TB/s

### Warp Reduction

Modern GPUs use warp shuffle:
```cuda
float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
```

## Exercises

1. **Basic reduction**: Naive then optimized
2. **Benchmark**: Compare versions, measure bandwidth
3. **Dot product**: Reduce after multiply

## Build & Run

```bash
./build.sh
./build/reduction
./build/dot_product
```
