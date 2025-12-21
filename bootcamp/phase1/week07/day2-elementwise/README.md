# Day 2: Element-wise Operations

## Learning Objectives

- Generalize the map pattern
- Implement various element-wise operations
- Understand arithmetic intensity

## Key Concepts

### The Map Pattern

Every element-wise operation follows the same pattern:
```cuda
output[i] = function(input[i])
// or
output[i] = function(input1[i], input2[i])
```

### Common Element-wise Operations

| Operation | Formula | Arithmetic Intensity |
|-----------|---------|---------------------|
| Copy | `out = in` | 0 FLOP / 8 bytes |
| Scale | `out = a * in` | 1 / 8 |
| Add | `out = a + b` | 1 / 12 |
| SAXPY | `out = a*x + y` | 2 / 12 |
| Square | `out = in * in` | 1 / 8 |
| Sqrt | `out = sqrt(in)` | 1 / 8 |
| Exp/Log | `out = exp(in)` | ~20 / 8 |

### Arithmetic Intensity

```
Arithmetic Intensity = FLOPs / Bytes Transferred
```

Higher intensity → more compute-bound
Lower intensity → more memory-bound

### Fusing Operations

Instead of:
```cuda
// Kernel 1: temp = a * x
// Kernel 2: y = temp + y
```

Combine into:
```cuda
// Single kernel: y = a * x + y
```

Reduces memory traffic by 50%!

## Exercises

1. **Element-wise operations**: Implement several and compare
2. **Fusion demo**: Show benefit of kernel fusion
3. **Compute vs memory bound**: Compare exp/log to simple ops

## Build & Run

```bash
./build.sh
./build/elementwise
./build/fusion_demo
```
