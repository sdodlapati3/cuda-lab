# Day 2: Tiled Convolution

## Learning Objectives
- Use shared memory for convolution halo regions
- Understand tile + halo loading patterns
- Measure speedup over naive approach

## Key Concepts

### Tiling Strategy
```
┌─────────────────────────────────┐
│  Halo   │    Tile Data    │Halo │
│ (extra) │ (output region) │     │
└─────────────────────────────────┘

Load (tile + 2*halo) into shared memory
Each thread can access neighbors without global memory
```

### Shared Memory Layout
- Tile size: 16x16 or 32x32 (output pixels)
- Halo: filter_radius on each side
- Total shared: (tile + 2*radius)² floats

### Loading Pattern
```
Block of 16x16 threads loads (16+2r)x(16+2r) region
Some threads load multiple pixels
Use modular indexing or thread remapping
```

## Performance Improvement
- Naive 3x3: Each pixel read 9x
- Tiled 3x3: Each pixel read ~1.2x (halo overlap between tiles)
- Expected: 4-7x speedup over naive
