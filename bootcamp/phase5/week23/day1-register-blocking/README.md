# Day 1: Register Blocking Introduction

## The Problem with Basic Tiling

In Week 22, each thread computed ONE output:
- Load 1 element from shared A
- Load 1 element from shared B  
- 1 FMA operation
- Repeat K times

Result: Low compute-to-memory ratio

## Register Blocking Solution

Each thread computes a TILE of outputs (e.g., 4×4 = 16 elements):
- Load 4 elements from shared A → 4 registers
- Load 4 elements from shared B → 4 registers
- Compute 4×4 = 16 FMA operations
- Much higher compute-to-memory ratio!

## Compute Intensity Analysis

### Without Register Blocking
- Loads per FMA: 2 (one from A, one from B)
- Compute intensity: 0.5 FMA/load

### With 4×4 Register Blocking
- Loads per iteration: 4 + 4 = 8
- FMAs per iteration: 4 × 4 = 16
- Compute intensity: 2.0 FMA/load (4× better!)

## Thread Block Organization

```
Block Tile: 128×128 outputs
Thread Tile: 8×8 outputs per thread
Threads per block: (128/8) × (128/8) = 16×16 = 256 threads
```

## Exercises
1. Calculate compute intensity for different thread tile sizes
2. Determine register requirements
3. Estimate occupancy impact
4. Plan implementation approach
