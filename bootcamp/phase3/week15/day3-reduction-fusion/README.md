# Day 3: Reduction Fusion

## Learning Objectives

- Fuse transform with reduction
- Implement fused dot product
- Fuse softmax (reduce + transform)

## Key Concepts

### Transform-Reduce Pattern

```cpp
// Separate: transform then reduce
transform<<<...>>>(x, temp);  // temp[i] = x[i]²
reduce<<<...>>>(temp, result);  // sum of temp

// Fused: transform in reduce
__global__ void fused_sum_squares(const float* x, float* result, int n) {
    // Each thread: load, square, contribute to block sum
    float val = x[idx];
    float squared = val * val;  // Transform in register
    
    // Now reduce squared (not temp[i])
    // ... reduction logic with squared
}
```

### Common Fused Reductions

| Operation | Formula | Fusion Benefit |
|-----------|---------|----------------|
| Sum of squares | Σx² | Skip temp array |
| L2 norm | √(Σx²) | Skip temp, add sqrt |
| Dot product | Σ(x·y) | Load 2 arrays, no temp |
| Cross entropy | -Σ(y·log(ŷ)) | Complex per-element |

### Softmax Fusion

```
Unfused:
  max = reduce_max(x)
  exp = elementwise(x - max)
  sum = reduce_sum(exp)
  out = elementwise(exp / sum)
  
Fused (2-pass):
  Pass 1: Compute max
  Pass 2: exp, sum, normalize (combined)
```

## Build & Run

```bash
./build.sh
./build/reduction_fusion
```
