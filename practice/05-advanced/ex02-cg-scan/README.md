# Exercise 02: CG Parallel Scan

## Objective
Implement parallel prefix sum (scan) using Cooperative Groups.

## Requirements
1. Implement both inclusive and exclusive scan
2. Use `cg::inclusive_scan()` and `cg::exclusive_scan()`
3. Handle block-sized arrays correctly

## Background
Scan (prefix sum) is a fundamental parallel primitive used in:
- Stream compaction
- Radix sort
- Histogram equalization
- Sparse matrix operations

**Inclusive scan**: Output[i] = Input[0] + Input[1] + ... + Input[i]
**Exclusive scan**: Output[i] = Input[0] + Input[1] + ... + Input[i-1]

## Testing
```bash
make
./test.sh
```
