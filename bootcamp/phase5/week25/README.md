# Week 25: Warp-Level GEMM

## Overview
This week explores warp-level optimizations using cooperative
loading patterns and warp shuffle operations.

## Daily Schedule

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 1 | Warp Organization | Warp-centric GEMM design |
| 2 | Cooperative Loading | Warp-wide data movement |
| 3 | Warp Shuffle | Data sharing within warp |
| 4 | Warp Tile | Each warp computes a tile |
| 5 | Multiple Warps | Block-level cooperation |
| 6 | Performance Tuning | Optimal warp configurations |

## Key Concept: Warp Tile

Instead of thread tile, use warp tile:
- Entire warp (32 threads) computes one output tile
- Better data sharing through registers
- Reduced shared memory traffic

```
Block Tile: 256×128
Warp Tile: 64×64
Thread Tile: 8×8
```

## Warp Shuffle Benefits
- Direct register-to-register communication
- No shared memory needed
- Very low latency

## Expected Performance
After this week: 60-75% of cuBLAS performance
