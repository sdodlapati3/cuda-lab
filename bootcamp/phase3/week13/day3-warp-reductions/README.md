# Day 3: Warp Reductions

## Learning Objectives

- Implement efficient warp-level reductions
- Compare shuffle vs shared memory approaches
- Handle various reduction operations

## Key Concepts

### Warp Reduction Pattern

```
Using __shfl_down_sync for reduction:

Initial: [0] [1] [2] [3] [4] [5] [6] [7] ... [31]

Step 1 (offset=16):
  Lane 0 += Lane 16, Lane 1 += Lane 17, ...
  
Step 2 (offset=8):
  Lane 0 += Lane 8, Lane 1 += Lane 9, ...
  
Step 3 (offset=4):
  Lane 0 += Lane 4, ...
  
Step 4 (offset=2):
  Lane 0 += Lane 2, ...
  
Step 5 (offset=1):
  Lane 0 += Lane 1
  
Result: Lane 0 has sum of all 32 values
```

### Generic Warp Reduce Template

```cpp
template<typename T, typename Op>
__device__ T warp_reduce(T val, Op op) {
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = op(val, other);
    }
    return val;  // Valid in lane 0
}
```

### Common Reduction Operations

| Operation | Function |
|-----------|----------|
| Sum | `val += other` |
| Max | `val = max(val, other)` |
| Min | `val = min(val, other)` |
| Product | `val *= other` |
| Bitwise OR | `val |= other` |

## Build & Run

```bash
./build.sh
./build/warp_reduce
```
