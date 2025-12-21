# Day 6: Warp-Level Patterns

## Learning Objectives

- Combine warp primitives for complex operations
- Build reusable warp-level building blocks
- Apply to real algorithms

## Key Concepts

### Pattern Library

This day covers common patterns built from warp primitives:

1. **Warp Broadcast** - Share value from one lane to all
2. **Warp Gather** - Collect values based on index
3. **Warp Segmented Reduce** - Reduce within segments
4. **Warp Match** - Find lanes with same value

### Building Blocks

```cpp
// Broadcast from lane 0
T warp_broadcast(T val) {
    return __shfl_sync(FULL_MASK, val, 0);
}

// Find first set bit (first active lane)
int warp_first_lane(unsigned int ballot) {
    return __ffs(ballot) - 1;
}

// Count lanes with matching value
int warp_count_equal(int val) {
    unsigned int matches = __match_any_sync(FULL_MASK, val);
    return __popc(matches);
}
```

### When to Use What

| Need | Pattern |
|------|---------|
| Sum all values | warp_reduce with __shfl_down |
| Prefix sum | warp_scan with __shfl_up |
| Any/All test | __any_sync / __all_sync |
| Find matches | __ballot_sync + __popc |
| Broadcast | __shfl_sync(val, src_lane) |

## Build & Run

```bash
./build.sh
./build/warp_patterns
```
