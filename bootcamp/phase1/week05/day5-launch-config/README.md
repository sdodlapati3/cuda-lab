# Day 5: Launch Configuration & Occupancy

## Learning Objectives

- Choose optimal block size and grid size
- Understand occupancy and its limits
- Use the occupancy calculator API

## Key Concepts

### Block Size Guidelines

| Block Size | Pros | Cons |
|------------|------|------|
| 64 | Low register pressure | May underutilize SM |
| 128 | Good balance | Safe default |
| 256 | Common choice | Higher register pressure |
| 512 | Max threads active | Very high register pressure |
| 1024 | Maximum allowed | Rarely optimal |

### Occupancy

**Occupancy** = Active warps / Max warps per SM

Limited by:
1. **Registers per thread**: More registers → fewer concurrent threads
2. **Shared memory per block**: More smem → fewer concurrent blocks
3. **Block size**: Threads per block affects scheduling

### Occupancy API

```cuda
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,    // Minimum grid size for max occupancy
    &blockSize,      // Suggested block size
    myKernel,        // Kernel function
    0,               // Dynamic shared memory
    0                // Block size limit (0 = no limit)
);
```

### Launch Bound Hints

```cuda
__global__ void __launch_bounds__(256, 4)
my_kernel() {
    // maxThreadsPerBlock = 256
    // minBlocksPerMultiprocessor = 4
}
```

## Exercises

1. **Find optimal block size**: Benchmark different sizes
2. **Use occupancy API**: Let CUDA choose block size
3. **Profile occupancy**: Use Nsight to see achieved occupancy

## Build & Run

```bash
./build.sh
./build/launch_config
./build/occupancy_demo
```
