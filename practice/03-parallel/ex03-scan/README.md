# Exercise: Prefix Sum (Scan)

## Objective
Implement parallel prefix sum using Hillis-Steele and Blelloch algorithms.

## Background

Prefix sum (scan) computes running totals:
- Input: [3, 1, 4, 1, 5, 9]
- Inclusive scan: [3, 4, 8, 9, 14, 23]
- Exclusive scan: [0, 3, 4, 8, 9, 14]

## Task

1. Implement Hillis-Steele scan (step-efficient)
2. Implement Blelloch scan (work-efficient)
3. Compare performance

## Algorithms

### Hillis-Steele
- O(n log n) work, O(log n) steps
- Simpler, good for small arrays

### Blelloch
- O(n) work, O(log n) steps
- Two phases: up-sweep, down-sweep
- Better for large arrays

## Files

- `scan.cu` - Skeleton
- `solution.cu` - Reference
- `Makefile` - Build
- `test.sh` - Test
