# Week 10: Occupancy Deep Dive

## Theme: Understanding and Optimizing GPU Occupancy

Occupancy is the ratio of active warps to maximum possible warps on an SM.
This week dives deep into what limits occupancy and how to optimize it.

## Daily Breakdown

| Day | Topic | Key Files |
|-----|-------|-----------|
| 1 | What is Occupancy? | occupancy_basics.cu |
| 2 | Register Pressure | register_analysis.cu |
| 3 | Shared Memory Limits | shared_mem_limits.cu |
| 4 | Block Size Selection | block_size_tuning.cu |
| 5 | Occupancy Calculator | calculator_demo.cu |
| 6 | When Occupancy Matters | occupancy_tradeoffs.cu |

## Key Mental Models

### Occupancy Formula
```
Occupancy = Active Warps per SM / Max Warps per SM

A100: Max 64 warps (2048 threads) per SM
```

### Limiting Factors
```
1. Registers: 65536 per SM → limits warps by usage
2. Shared Memory: 164KB per SM → limits blocks
3. Block Size: Max 1024 threads → warps per block
4. Block Limit: Max 32 blocks per SM
```

### The Occupancy Lie
Higher occupancy ≠ always better performance!
- Sometimes lower occupancy = more resources per thread
- The key is hiding latency, not maximizing threads

## Prerequisites
- Week 9: Roofline Model (understand compute vs memory bound)
- Phase 1: Shared memory and block structure

## Building
Each day has its own build.sh script:
```bash
cd dayX-topic
./build.sh
./build/executable_name
```
