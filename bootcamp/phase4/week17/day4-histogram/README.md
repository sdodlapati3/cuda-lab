# Day 4: Histogram & Equalization

## Learning Objectives
- Implement atomic-based histogram on GPU
- Understand histogram equalization algorithm
- Reduce atomic contention with privatization

## Key Concepts

### Histogram Algorithm
```
For each pixel:
    bin = pixel_value / bin_width
    atomicAdd(&histogram[bin], 1)
```

### Atomic Contention
- Many threads updating same bins → serialization
- Solution: Private histograms per block, then reduce

### Histogram Equalization
```
1. Compute histogram
2. Compute CDF (cumulative sum)
3. Normalize: new_value = CDF[old_value] * 255 / total_pixels
```

## Implementation Strategy
1. **Global atomics**: Simple but slow
2. **Shared memory histogram**: Private per block
3. **Warp-level aggregation**: Reduce atomics further
