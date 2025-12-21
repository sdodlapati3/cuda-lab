# Day 4: Scan (Prefix Sum)

## Learning Objectives

- Implement inclusive and exclusive scan
- Understand scan algorithms and applications
- Parallel prefix pattern

## Key Concepts

### What is Scan?

Transform array using cumulative operation:

**Exclusive scan**: `out[i] = sum(in[0..i-1])`
```
Input:  [3, 1, 7, 0, 4, 1, 6, 3]
Output: [0, 3, 4, 11, 11, 15, 16, 22]
```

**Inclusive scan**: `out[i] = sum(in[0..i])`
```
Input:  [3, 1, 7, 0, 4, 1, 6, 3]
Output: [3, 4, 11, 11, 15, 16, 22, 25]
```

### Applications

| Application | Use of Scan |
|-------------|-------------|
| Stream compaction | Build output indices |
| Radix sort | Count positions |
| Polynomial evaluation | Horner's method parallel |
| Sparse matrix | Row pointer array |
| Histogram equalization | CDF computation |

### Algorithms

1. **Hillis-Steele**: Work-inefficient but low span
2. **Blelloch**: Work-efficient, two-phase (up-sweep, down-sweep)
3. **Warp-based**: Use shuffle for small arrays
4. **Decoupled look-back**: Modern GPU approach (CUB uses this)

### Work Complexity

- Sequential: O(n) work
- Naive parallel: O(n log n) work
- Blelloch: O(n) work, O(log n) span

## Exercises

1. **Warp scan**: Single warp prefix sum
2. **Block scan**: Shared memory approach
3. **Large array scan**: Multi-block strategy

## Build & Run

```bash
./build.sh
./build/scan_demo
./build/stream_compact
```
