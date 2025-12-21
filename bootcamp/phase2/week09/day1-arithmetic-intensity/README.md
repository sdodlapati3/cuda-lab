# Day 1: Arithmetic Intensity

## Learning Objectives

- Define arithmetic intensity precisely
- Calculate AI for common kernels
- Understand operational vs memory intensity

## Key Concepts

### Definition

```
Arithmetic Intensity (AI) = FLOPs / Bytes Transferred

Units: FLOP / Byte (or FLOPS/s / Bytes/s = dimensionless)
```

### Why It Matters

AI determines whether your kernel is:
- **Memory-bound** (low AI): Performance limited by data movement
- **Compute-bound** (high AI): Performance limited by arithmetic units

### Calculating AI

**Step 1:** Count floating-point operations
- `+`, `-`, `*`, `/` each = 1 FLOP
- `fma(a,b,c) = a*b+c` = 2 FLOPs
- Transcendentals (sin, exp) = varies (check docs)

**Step 2:** Count bytes transferred
- Each `float` load = 4 bytes
- Each `float` store = 4 bytes
- Count unique accesses (caching matters!)

**Step 3:** Divide

### Examples

| Kernel | FLOPs | Bytes | AI |
|--------|-------|-------|-------|
| Vector Add: C[i] = A[i] + B[i] | 1 | 12 (read 2, write 1) | 0.083 |
| SAXPY: Y[i] = a*X[i] + Y[i] | 2 | 12 | 0.167 |
| Reduction: sum += A[i] | 1 | 4 | 0.25 |
| Dot Product: sum += A[i]*B[i] | 2 | 8 | 0.25 |
| MatMul (naive): C[i][j] += A[i][k]*B[k][j] | 2N | 4N+4N+4 | ~N/4 |

### Operational vs Data Intensity

**Operational Intensity:** AI based on actual bytes moved
**Data Intensity:** AI based on problem size (ignoring cache)

For memory-bound analysis, operational intensity matters.

## Exercises

1. Calculate AI for your Phase 1 vector_add kernel
2. Calculate AI for your reduction kernel
3. Predict which Phase 1 kernels are memory-bound

## Build & Run

```bash
./build.sh
./build/ai_calculator
```
