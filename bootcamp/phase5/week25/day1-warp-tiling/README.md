# Week 25, Day 1: Warp Tiling Introduction

## Objective
Understand warp-level execution and design GEMM with warp tiles.

## Key Concepts
- Warp: 32 threads executing in lockstep
- No explicit synchronization needed within warp
- Warp tile: portion of output computed by one warp
- Implicit coordination through shuffle operations

## Why Warp Tiling?
- Threads in warp can share data via registers
- No shared memory needed for intra-warp communication
- Reduces synchronization points
- Better register utilization

## Warp Tile Design
- Block tile: 128×128 (computed by thread block)
- Warp tile: 32×64 (computed by one warp)
- Thread tile: 8×8 (computed by one thread)
- 4 warps per block (128/32 × 128/64)

## Expected Results
- Foundation for shuffle-based optimizations
- Similar performance, cleaner architecture
