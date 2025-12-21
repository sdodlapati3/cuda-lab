# Day 6: Fusion Strategies

## Learning Objectives

- Know when to fuse and when not to
- Understand fusion anti-patterns
- Make informed fusion decisions

## Key Concepts

### When to Fuse ✅

1. **Consecutive element-wise operations**
   - Always fuse: add→mul→activation
   - Memory traffic reduction is proportional

2. **Transform + Reduce**
   - Square → sum = sum of squares
   - Multiply → sum = dot product

3. **Operations with same data access pattern**
   - Same indices, same order

### When NOT to Fuse ❌

1. **Register pressure too high**
   - Fusing spills to local memory → worse performance
   - Check with `--ptxas-options=-v`

2. **Different parallelization strategies needed**
   - One op needs 1D grid, other needs 2D
   - One is compute-bound, other memory-bound

3. **Debugging complexity**
   - Fused kernels harder to profile
   - Keep unfused version for comparison

4. **Diminishing returns**
   - Already compute-bound? Fusion won't help
   - Last operation is I/O? Still bottlenecked

### Decision Framework

```
Is the next operation on the same data?
  └─ No → Don't fuse
  └─ Yes → Are both memory-bound?
              └─ No → Consider keeping separate
              └─ Yes → Will registers spill?
                         └─ Yes → Profile first
                         └─ No → FUSE!
```

## Build & Run

```bash
./build.sh
./build/fusion_strategies
```
