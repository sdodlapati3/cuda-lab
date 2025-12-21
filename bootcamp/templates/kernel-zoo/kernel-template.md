# Kernel Zoo: Kernel Documentation Template

## Kernel Name: [e.g., Warp-Shuffle Reduction]

**Category:** Primitives / GEMM / ML Ops  
**Difficulty:** ⭐ Easy / ⭐⭐ Medium / ⭐⭐⭐ Advanced  
**Phase:** [When to learn this]

---

## Problem Statement

**Input:** Describe the input (e.g., array of N floats)  
**Output:** Describe the output (e.g., single sum value)  
**Constraints:** Any size/alignment requirements

---

## The Journey

### Version 1: Naive Implementation

```cuda
// naive.cu
__global__ void reduce_naive(const float* input, float* output, int n) {
    // Implementation
}
```

**Performance:** X GB/s (Y% of peak)  
**Bottleneck:** [What limits performance]

### Version 2: [Optimization Name]

```cuda
// optimized.cu
__global__ void reduce_optimized(...) {
    // Implementation
}
```

**Performance:** X GB/s (Y% of peak)  
**Key Insight:** [What made it faster]

### Version 3: [Further Optimization]

```cuda
// final.cu
__global__ void reduce_final(...) {
    // Implementation
}
```

**Performance:** X GB/s (Y% of peak)  
**Key Insight:** [What made it faster]

---

## Performance Comparison

| Version | Time (μs) | GB/s | % Peak | Speedup vs Naive |
|---------|-----------|------|--------|------------------|
| Naive   | 524.3     | 122  | 12.7%  | 1.00×           |
| V2      | 203.7     | 314  | 32.7%  | 2.57×           |
| V3      | 82.1      | 780  | 81.2%  | 6.39×           |
| CUB     | 79.2      | 807  | 84.0%  | 6.62×           |

**Roofline Position:** Memory-bound / Compute-bound

---

## Key Optimizations Applied

### Optimization 1: [Name]
**Technique:** [What you did]  
**Impact:** X% speedup  
**Trade-off:** [Any downsides]

### Optimization 2: [Name]
**Technique:** [What you did]  
**Impact:** X% speedup  
**Trade-off:** [Any downsides]

---

## Profiler Evidence

```
Metric                          Naive       Optimized
─────────────────────────────────────────────────────
Memory Throughput (%)           15.2        81.4
Compute Throughput (%)          8.3         12.1
Achieved Occupancy              0.45        0.72
L2 Cache Hit Rate               12.5%       45.3%
```

**Screenshot:** [Link to Nsight Compute screenshot if helpful]

---

## Lessons Learned

1. **[Lesson 1]:** What surprised you
2. **[Lesson 2]:** What you'd do differently
3. **[Lesson 3]:** How this applies elsewhere

---

## Files

```
kernel-zoo/primitives/reduction/
├── naive.cu              # First implementation
├── warp_shuffle.cu       # Warp-level optimization
├── block_reduce.cu       # Block-level with shared memory
├── multi_block.cu        # Multi-block reduction
├── benchmark.cu          # Performance comparison
├── test.cu               # Correctness tests
├── cpu_reference.cpp     # CPU baseline for verification
└── README.md             # This file
```

---

## References

- [NVIDIA Reduction Optimization](https://developer.nvidia.com/blog/optimizing-parallel-reduction-cuda/)
- [CUB Library](https://nvlabs.github.io/cub/)
- [Related paper if applicable]

---

## TODO

- [ ] Test on different GPU architectures
- [ ] Add FP16/BF16 variants
- [ ] Profile on larger data sizes
- [ ] Compare with Triton implementation
