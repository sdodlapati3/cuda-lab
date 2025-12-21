# Week 23: Register Blocking & Memory Hierarchy

## Overview
This week focuses on register-level optimizations to dramatically
improve GEMM performance. Each thread will compute multiple outputs.

## Daily Schedule

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 1 | Register Blocking Intro | Each thread computes MxN outputs |
| 2 | 2x2 Thread Tile | Basic register blocking |
| 3 | 4x4 Thread Tile | Increased compute per thread |
| 4 | Register Pressure | Balancing registers and occupancy |
| 5 | Memory Hierarchy | L1/L2 cache optimization |
| 6 | Performance Analysis | Comprehensive benchmark |

## Key Concept: Thread Tile

Instead of: 1 thread → 1 output element
Now: 1 thread → MxN output elements

Benefits:
- Better instruction-level parallelism
- Reduced shared memory traffic per output
- Higher compute intensity

## Expected Performance
After this week: 50-70% of cuBLAS performance
Target: 1.5-2x improvement over Week 22
