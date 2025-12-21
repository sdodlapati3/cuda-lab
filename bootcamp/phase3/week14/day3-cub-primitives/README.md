# Day 3: CUB Primitives

## Learning Objectives

- Use CUB device-wide algorithms
- Understand CUB's temporary storage pattern
- Compare CUB to custom implementations

## Key Concepts

### What is CUB?

CUB (CUDA UnBound) provides:
- **Device-wide** algorithms (reduce, scan, sort, select)
- **Block-level** cooperative primitives
- **Warp-level** primitives
- Optimal performance with minimal code

### CUB Usage Pattern

```cpp
#include <cub/cub.cuh>

// Step 1: Determine temporary storage size
size_t temp_bytes = 0;
cub::DeviceReduce::Sum(nullptr, temp_bytes, d_in, d_out, n);

// Step 2: Allocate temporary storage
void* d_temp = nullptr;
cudaMalloc(&d_temp, temp_bytes);

// Step 3: Run the algorithm
cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
```

### Key Device Algorithms

| Algorithm | Function |
|-----------|----------|
| Reduce | `DeviceReduce::Sum`, `Max`, `Min` |
| Scan | `DeviceScan::ExclusiveSum`, `InclusiveSum` |
| Sort | `DeviceRadixSort::SortKeys`, `SortPairs` |
| Select | `DeviceSelect::If`, `Unique`, `Flagged` |
| Histogram | `DeviceHistogram::HistogramEven` |

## Build & Run

```bash
./build.sh
./build/cub_basics
```
