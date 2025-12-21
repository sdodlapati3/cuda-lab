# Day 1: 2D Tiling Introduction

## Learning Objectives
- Understand why 2D tiling is essential for GEMM
- Derive data reuse factor mathematically
- Visualize tile movement through matrices
- Plan implementation approach

## The Key Insight

### Problem with 1D Tiling
- Row-tiling: A reused, B read M×N×K times
- Col-tiling: B reused, A read M×N×K times
- Neither alone solves the problem

### 2D Tiling Solution
Tile BOTH matrices simultaneously:
- Load tile of A (TILE × K elements)
- Load tile of B (K × TILE elements)
- Compute TILE × TILE output elements

## Algorithm Overview

```
for each output tile (TILE_M × TILE_N):
    initialize accumulators to 0
    
    for k = 0 to K in steps of TILE_K:
        1. Load A tile (TILE_M × TILE_K) → shared_A
        2. Load B tile (TILE_K × TILE_N) → shared_B
        3. Synchronize
        4. Multiply-accumulate from shared memory
        5. Synchronize
    
    Write TILE_M × TILE_N results to C
```

## Data Reuse Analysis

### Without Tiling
- A read M × N times (once per output column)
- B read M × N times (once per output row)
- Total: 2 × M × N × K reads

### With Tiling
- A read M/TILE × N/TILE × TILE × K = M × N × K / TILE times
- B read similarly reduced
- Reduction factor: **TILE**

For TILE = 32: 32× fewer global memory reads!

## Memory Traffic

### Naive GEMM (M=N=K=1024)
- Global reads: 2 × 1024³ = 2 billion reads
- Completely memory-bound

### Tiled GEMM (TILE=32)
- Global reads: 2 × 1024³ / 32 = 64 million reads
- 32× improvement

## Visualization

```
A (M × K)                    B (K × N)
┌───────────────────┐        ┌───────────────────┐
│   │   │   │   │   │        │ ░ │   │   │   │   │
│   │   │   │   │   │        │ ░ │   │   │   │   │
│───┼───┼───┼───┼───│        │ ░ │   │   │   │   │
│ ░░░░░░░░░░░░░░░░░ │   ×    │ ░ │   │   │   │   │
│───┼───┼───┼───┼───│        │ ░ │   │   │   │   │
│   │   │   │   │   │        │───┼───┼───┼───┼───│
└───────────────────┘        └───────────────────┘
    Row of tiles                Col of tiles
                ↓
        ┌───┐
        │ █ │  = Output tile (TILE × TILE)
        └───┘
```

## Shared Memory Layout

```cpp
__shared__ float As[TILE_M][TILE_K];  // Tile of A
__shared__ float Bs[TILE_K][TILE_N];  // Tile of B

// Each thread block computes one TILE_M × TILE_N output tile
// Iterates K/TILE_K times, loading new tiles each iteration
```

## Exercises
1. Calculate data reuse for different tile sizes
2. Determine shared memory requirements
3. Estimate occupancy impact
4. Draw diagram of tile movement
