# Day 2: Tiled N-Body Simulation

## Learning Objectives
- Understand memory-bound nature of naive N-body
- Implement shared memory tiling for body data
- Analyze tile size impact on performance
- Achieve significant speedup over naive approach

## Key Concepts

### Why Tiling?
The naive N-body approach loads N bodies from global memory for each of N bodies:
- N² global memory loads
- Global memory bandwidth: ~1-2 TB/s
- This becomes the bottleneck, not compute

### Tiled Algorithm
1. Each thread block loads a tile of bodies to shared memory
2. All threads compute interactions with the tile
3. Repeat for all tiles
4. Global loads reduced by factor of block size!

### Memory Access Pattern
```
For each tile t = 0 to numTiles:
    1. Each thread loads one body to shared memory
    2. __syncthreads()
    3. Each thread computes interactions with all bodies in tile
    4. __syncthreads()
    5. Accumulate partial forces
```

## Performance Analysis

| Metric | Naive | Tiled (256 threads) |
|--------|-------|---------------------|
| Global loads | N² | N²/256 |
| Bandwidth utilization | Low | High |
| Compute utilization | Low | High |
| Expected speedup | 1x | 5-10x |

## Exercises
1. Implement tiled N-body kernel
2. Experiment with different tile/block sizes
3. Compare performance with naive version
4. Profile memory throughput with nsys
