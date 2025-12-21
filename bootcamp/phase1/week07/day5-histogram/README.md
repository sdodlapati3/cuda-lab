# Day 5: Histogram

## Learning Objectives

- Implement histogram: the "scatter" pattern
- Master atomic operations
- Optimization techniques: privatization, shared memory

## Key Concepts

### What is Histogram?

Count occurrences in each bin:
```cuda
for (int i = 0; i < n; i++) {
    int bin = compute_bin(input[i]);
    histogram[bin]++;
}
```

### The Challenge

Multiple threads may update same bin → race condition!

Solution: Atomic operations

### Atomic Operations

```cuda
atomicAdd(&histogram[bin], 1);  // Thread-safe increment
```

But atomics are slow when many threads hit same bin!

### Optimization Strategies

| Strategy | Description | Benefit |
|----------|-------------|---------|
| Shared memory | Private histogram per block | Fewer global atomics |
| Privatization | Multiple private histograms | No conflicts |
| Replication | Thread-private then merge | No atomics during count |

### Performance Considerations

- Few bins + many elements → high contention
- Many bins + sparse data → random access
- Sweet spot: shared memory with block-level aggregation

## Exercises

1. **Global atomic**: Simple but slow
2. **Shared memory**: Block-private histograms  
3. **Compare**: Measure improvement

## Build & Run

```bash
./build.sh
./build/histogram
```
