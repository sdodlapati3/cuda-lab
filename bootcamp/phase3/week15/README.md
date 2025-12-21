# Week 15: Kernel Fusion

Eliminate kernel launch overhead and memory traffic by combining operations.

## Daily Breakdown

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Fusion Basics | Why fuse, benefits, simple examples |
| 2 | Element-wise Fusion | Combine transforms, reduce memory traffic |
| 3 | Reduction Fusion | Fuse transform + reduce |
| 4 | Tiled Fusion | Fuse operations in shared memory |
| 5 | Producer-Consumer | Chain operations in registers |
| 6 | Fusion Strategies | When to fuse, when not to |

## Mental Model

```
Separate Kernels:           Fused Kernel:
┌─────────┐                 ┌─────────────────┐
│ Kernel A│→ Global Mem →   │ A + B + C fused │
├─────────┤                 │ (stays in regs) │
│ Kernel B│→ Global Mem →   └─────────────────┘
├─────────┤                         ↓
│ Kernel C│                    1 memory trip
└─────────┘
    ↓
3 memory trips
```

## Key Metrics

- **Kernel launches saved**: N kernels → 1 kernel
- **Memory traffic reduction**: Often 2-10x less
- **Register pressure**: Fused kernels use more registers
- **Occupancy trade-off**: More registers = fewer threads

## Quick Reference

```cpp
// Before: 2 kernels, 4 memory transactions
square<<<...>>>(in, temp);
add<<<...>>>(temp, bias, out);

// After: 1 kernel, 2 memory transactions
__global__ void fused(float* in, float* bias, float* out) {
    float x = in[idx];
    out[idx] = x * x + bias[idx];  // Stays in register
}
```
