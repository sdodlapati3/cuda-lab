# Day 3: Memory Access Analysis

## Learning Objectives
- Analyze memory access patterns in naive GEMM
- Understand coalescing behavior for A and B matrices
- Quantify memory bandwidth utilization
- Identify optimization opportunities

## Memory Access Patterns

### Matrix A Access
```
Thread (row, col) reads: A[row][0], A[row][1], ..., A[row][K-1]

For threads in same warp (32 consecutive threads):
- Warp has 32 threads with consecutive col values
- All threads read same row index initially
- Strided access across K dimension

Pattern: A[row * K + k] for all threads in warp
         Different rows → NOT coalesced!
```

### Matrix B Access
```
Thread (row, col) reads: B[0][col], B[1][col], ..., B[K-1][col]

For threads in same warp:
- 32 threads have consecutive col values
- Each thread reads different column
- Access: B[k * N + col], B[k * N + col+1], ..., B[k * N + col+31]

Pattern: Consecutive memory addresses → Coalesced!
```

## Coalescing Analysis

### Ideal Case
32 threads access 32 consecutive 4-byte values = 128 bytes
One memory transaction

### Naive GEMM Reality
- **Matrix A**: 32 threads access same element (broadcast) or strided
- **Matrix B**: 32 threads access consecutive elements ✓
- Result: ~50% coalescing efficiency at best

## Memory Traffic Analysis

### Theoretical Minimum
```
Read A: M × K × 4 bytes (once)
Read B: K × N × 4 bytes (once)
Write C: M × N × 4 bytes
Total: 4(MK + KN + MN) bytes
```

### Naive Actual
```
Each output element reads:
- K elements from A
- K elements from B

Total reads: M × N × K × 2 × 4 bytes
Amplification: ~K× more than optimal
```

## Profiling Commands
```bash
# Memory throughput
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum ./naive_gemm

# Memory efficiency  
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./naive_gemm

# Full memory analysis
ncu --set memory ./naive_gemm
```

## Exercises
1. Profile naive GEMM and measure memory throughput
2. Calculate achieved bandwidth vs peak
3. Identify coalescing efficiency from profiler
4. Explain why A access is inefficient
