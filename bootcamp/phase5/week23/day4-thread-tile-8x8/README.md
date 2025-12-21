# Week 23, Day 4: 8×8 Thread Tile

## Objective
Push register blocking to 8×8 outputs per thread - 64 accumulators.

## Key Concepts
- Maximum practical thread tile size
- Significant register pressure (100+ registers)
- Near-optimal compute intensity
- Trade-off between parallelism and work per thread

## Register Analysis
- c[8][8] = 64 floats → 64 registers
- regA[8], regB[8] = 16 registers
- Total: 80+ registers per thread
- A100 has 255 max registers per thread

## Compute Intensity
- 8+8 = 16 loads from shared memory
- 64 FMAs computed
- Intensity: 64/16 = 4.0 FMA/load

## Expected Results
- 40-50% of cuBLAS
- May see diminishing returns due to occupancy
