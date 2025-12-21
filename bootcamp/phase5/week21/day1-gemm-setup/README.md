# Day 1: GEMM Problem Setup

## Learning Objectives
- Understand GEMM dimensions and notation
- Calculate FLOP counts and arithmetic intensity
- Apply the roofline model to predict performance
- Set up matrix storage and initialization

## GEMM Definition

### Mathematical Form
```
C[M,N] = α × A[M,K] × B[K,N] + β × C[M,N]
```

### Common Conventions
- **M**: Number of rows in A and C
- **N**: Number of columns in B and C
- **K**: Shared dimension (columns of A, rows of B)
- **α**: Scalar multiplier for A×B
- **β**: Scalar multiplier for existing C

### Transpose Variants
| Variant | A | B | Description |
|---------|---|---|-------------|
| NN | Normal | Normal | A × B |
| NT | Normal | Transposed | A × Bᵀ |
| TN | Transposed | Normal | Aᵀ × B |
| TT | Transposed | Transposed | Aᵀ × Bᵀ |

## Performance Metrics

### FLOP Count
```
Total FLOPs = 2 × M × N × K
- M × N × K multiplications
- M × N × K additions (accumulate)
```

### Memory Traffic
```
Read A:  M × K elements
Read B:  K × N elements
Read C:  M × N elements (if β ≠ 0)
Write C: M × N elements

Total bytes (FP32): 4 × (M×K + K×N + 2×M×N)
```

### Arithmetic Intensity
```
AI = FLOPs / Bytes
   = 2×M×N×K / (4 × (M×K + K×N + 2×M×N))
   
For square matrices (M=N=K):
AI = 2N³ / (4 × 4N²) = N/8

For N=4096: AI = 512 FLOPs/byte
```

## Roofline Model

```
Achievable Performance = min(Peak FLOPS, Peak BW × AI)

A100 GPU:
- Peak FP32: 19.5 TFLOPS
- Peak BW: 2039 GB/s
- Ridge point: 19.5e12 / 2039e9 = 9.6 FLOPs/byte

For GEMM with AI > 9.6: Compute-bound (good!)
```

## Exercises
1. Calculate FLOP count for M=4096, N=4096, K=4096
2. Calculate arithmetic intensity for same dimensions
3. Use roofline model to predict max achievable TFLOPS
4. Implement matrix initialization with random values
