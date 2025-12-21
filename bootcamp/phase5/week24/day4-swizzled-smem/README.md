# Week 24, Day 4: Swizzled Shared Memory

## Objective
Eliminate bank conflicts through memory swizzling patterns.

## Key Concepts
- Shared memory bank conflicts review
- XOR-based swizzle patterns
- Conflict-free access for any thread pattern
- Integration with vectorized loads

## Why Swizzle?
- Standard padding wastes shared memory
- Swizzling remaps addresses without waste
- Works for any access pattern
- Used in high-performance libraries like CUTLASS

## XOR Swizzle Pattern
```cpp
int swizzled_col = col ^ (row & 0x3);  // Example 4-way swizzle
```

## Expected Results
- Zero bank conflicts
- Similar performance to padding, less memory used
