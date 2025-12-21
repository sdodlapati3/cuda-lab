# Week 23, Day 5: Register Pressure Analysis

## Objective
Understand register pressure and its impact on GEMM performance.

## Key Concepts
- Register file capacity (255 max per thread on A100)
- Occupancy vs registers per thread trade-off
- PTX analysis for register usage
- Finding optimal thread tile size

## Register Pressure Effects
- More registers → fewer warps → less latency hiding
- Fewer registers → more warps → more memory pressure
- Sweet spot depends on memory vs compute balance

## Analysis Tools
- nvcc --ptxas-options=-v (register count)
- Nsight Compute occupancy analysis
- cudaFuncGetAttributes for runtime info

## Expected Findings
- 4×4 tile: Good balance of registers and occupancy
- 8×8 tile: May spill or reduce occupancy
