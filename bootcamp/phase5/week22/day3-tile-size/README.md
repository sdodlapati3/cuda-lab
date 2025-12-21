# Day 3: Tile Size Selection

## Learning Objectives
- Understand factors affecting optimal tile size
- Experiment with different tile dimensions
- Analyze trade-offs between occupancy and reuse
- Choose optimal tiles for different matrix sizes

## Tile Size Trade-offs

### Larger Tiles
**Advantages:**
- More data reuse (fewer global memory reads)
- Better arithmetic intensity
- Fewer loop iterations

**Disadvantages:**
- More shared memory per block
- Lower occupancy (fewer concurrent blocks)
- Larger register pressure

### Smaller Tiles
**Advantages:**
- Higher occupancy
- Less shared memory usage
- Works for smaller matrices

**Disadvantages:**
- Less data reuse
- More global memory traffic
- More synchronization overhead

## Occupancy Analysis

For A100 (164KB shared memory per SM, max 2048 threads/SM):

| Tile Size | Shared Mem | Threads | Max Blocks/SM | Occupancy |
|-----------|------------|---------|---------------|-----------|
| 8×8       | 0.5 KB     | 64      | 32 (limited)  | ~50%      |
| 16×16     | 2 KB       | 256     | 8             | 100%      |
| 32×32     | 8 KB       | 1024    | 2             | 100%      |
| 64×64     | 32 KB      | 4096→1024| 1            | 50%       |

## Non-Square Tiles

Sometimes rectangular tiles perform better:
- 16×32: Better for certain access patterns
- 32×16: May improve coalescing
- Depends on matrix dimensions and GPU architecture

## Exercises
1. Benchmark tile sizes from 8×8 to 64×64
2. Find optimal tile size for different matrix sizes
3. Try non-square tiles
4. Profile with Nsight Compute to understand bottlenecks
