# Practice: Parallel Patterns

Hands-on exercises for fundamental parallel algorithms on GPU.

## Exercises

| Exercise | Topic | Difficulty | Week |
|----------|-------|------------|------|
| [ex01-reduction](ex01-reduction/) | Tree reduction patterns | ⭐⭐ | Week 4 |
| [ex02-warp-reduction](ex02-warp-reduction/) | Warp shuffle reduction | ⭐⭐ | Week 4 |
| [ex03-scan](ex03-scan/) | Prefix sum algorithms | ⭐⭐⭐ | Week 5 |
| [ex04-histogram](ex04-histogram/) | Parallel histogram | ⭐⭐⭐ | Week 4 |

## Learning Objectives

After completing these exercises, you will be able to:

1. **Implement parallel reduction** - Sum, min, max across millions of elements
2. **Use warp-level primitives** - `__shfl_down_sync`, `__ballot_sync`, etc.
3. **Build prefix sum** - Both inclusive and exclusive scan
4. **Handle race conditions** - Use atomics correctly for histogram

## Key Concepts

### Reduction
- Tree-based parallel summation
- Work-efficient vs step-efficient tradeoffs
- Warp divergence elimination

### Warp Primitives
- `__shfl_down_sync()` for warp-level reduction
- `__ballot_sync()` for voting
- No shared memory needed within warp

### Scan (Prefix Sum)
- Hillis-Steele: Step-efficient, more work
- Blelloch: Work-efficient, more steps
- Applications: stream compaction, radix sort

### Histogram
- Atomic increments with privatization
- Shared memory histogram for efficiency
- Final merge to global histogram
