# Exercise: Memory Coalescing

## Objective
Understand and implement coalesced memory access patterns to maximize GPU memory bandwidth.

## Background

Memory coalescing is one of the most important optimization techniques in CUDA. When threads in a warp access consecutive memory locations, the GPU can combine these accesses into fewer memory transactions.

### Coalesced Access (Good)
```
Thread 0 → Address 0
Thread 1 → Address 4
Thread 2 → Address 8
...
Thread 31 → Address 124
```
→ Single 128-byte transaction

### Strided Access (Bad)
```
Thread 0 → Address 0
Thread 1 → Address 128
Thread 2 → Address 256
...
```
→ 32 separate transactions!

## Task

1. **Implement `copy_coalesced`** - Copy array with coalesced access pattern
2. **Implement `copy_strided`** - Copy with strided access (intentionally bad)
3. **Implement `aos_to_soa`** - Convert Array of Structures to Structure of Arrays
4. **Measure and compare** performance

## Files

- `coalescing.cu` - Skeleton with TODOs
- `solution.cu` - Reference implementation
- `Makefile` - Build configuration
- `test.sh` - Validation script

## Expected Results

You should observe:
- Coalesced copy: ~80-90% of theoretical bandwidth
- Strided copy: ~10-20% of theoretical bandwidth
- AoS to SoA transformation significantly improves subsequent access patterns

## Hints

1. Use `threadIdx.x + blockIdx.x * blockDim.x` for coalesced indexing
2. For strided access, multiply index by stride
3. For AoS→SoA, each thread handles one element, writing to different output arrays

## Compilation

```bash
make
./coalescing
```

## Success Criteria

- [ ] Coalesced copy achieves >300 GB/s on modern GPU
- [ ] Strided copy shows >5x slowdown vs coalesced
- [ ] AoS to SoA transformation works correctly
- [ ] Understand WHY the performance differs
