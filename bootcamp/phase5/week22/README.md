# Week 22: Tiling Strategies

## Overview
This week introduces the critical concept of 2D tiling for GEMM optimization.
By tiling both A and B matrices, we achieve significant data reuse and
dramatically reduce global memory traffic.

## Daily Schedule

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 1 | 2D Tiling Introduction | Tile both A and B in shared memory |
| 2 | Basic Tiled GEMM | Implement TILE_SIZE × TILE_SIZE blocking |
| 3 | Tile Size Selection | Impact of tile dimensions on performance |
| 4 | Bank Conflicts | Shared memory access optimization |
| 5 | Double Buffering | Hide memory latency |
| 6 | Tiled Performance | Comprehensive benchmark |

## Learning Objectives
- Understand 2D tiling algorithm
- Implement shared memory tiled GEMM
- Choose optimal tile sizes
- Avoid shared memory bank conflicts
- Use double buffering for latency hiding

## Expected Performance
After this week: 20-40% of cuBLAS performance
Target: 5-10x improvement over Week 21

## Prerequisites
- Week 21: GEMM Fundamentals
- Understanding of shared memory
- Memory coalescing concepts
