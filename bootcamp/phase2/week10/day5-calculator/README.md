# Day 5: Occupancy Calculator

## Learning Objectives

- Use NVIDIA Occupancy Calculator
- Programmatically calculate occupancy
- Understand multi-factor limiting

## Key Concepts

### The Occupancy Equation

```
Max Blocks = min(
    max_blocks_by_threads,
    max_blocks_by_registers, 
    max_blocks_by_shared_mem,
    max_blocks_per_sm_limit
)

Active Warps = Max Blocks × Warps per Block
Occupancy = Active Warps / Max Warps per SM
```

### Using the Calculator

NVIDIA provides:
1. Excel spreadsheet calculator
2. Runtime API: `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
3. NSight Compute occupancy section

### Identifying the Limiter

```cpp
cudaFuncAttributes attr;
cudaFuncGetAttributes(&attr, kernel);
// Check attr.numRegs, attr.sharedSizeBytes
```

## Build & Run

```bash
./build.sh
./build/calculator_demo
```
