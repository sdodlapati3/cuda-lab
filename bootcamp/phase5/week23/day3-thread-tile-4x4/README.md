# Week 23, Day 3: 4×4 Thread Tile

## Objective
Expand register blocking to 4×4 outputs per thread for 4× compute intensity.

## Key Concepts
- 16 accumulators per thread
- 8 register fragments (4 for A, 4 for B)
- Outer product pattern scales naturally
- Managing register pressure

## Register Usage
- c[4][4] = 16 floats → 16 registers for accumulators
- regA[4], regB[4] = 8 registers for fragments
- Total: ~24+ registers per thread

## Compute Intensity
- 4 loads from As + 4 loads from Bs = 8 loads
- 16 FMAs computed
- Intensity: 16/8 = 2.0 FMA/load

## Expected Results
- 30-40% of cuBLAS
- ~2× speedup over 2×2 tile
