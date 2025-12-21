# Day 6: When Occupancy Matters

## Learning Objectives

- Understand when high occupancy helps
- Learn when low occupancy is acceptable
- See the occupancy-performance relationship

## Key Concepts

### The Occupancy Myth

**More occupancy ≠ always faster!**

Occupancy is about **latency hiding**, not raw performance:
- Memory-bound: high occupancy helps hide latency
- Compute-bound: may not need high occupancy
- Cache-friendly: lower occupancy can be better

### When High Occupancy Helps

- Memory-bound kernels (waiting for DRAM)
- Kernels with irregular memory access
- Kernels with synchronization barriers

### When Low Occupancy is OK

- Compute-bound (already saturated)
- High cache reuse (tiles in L1/smem)
- Sequential memory access (no stalls)

### The Trade-off

```
Higher occupancy often means:
- Fewer registers per thread → more spilling
- Less shared memory per block → smaller tiles
- Potentially SLOWER due to resource pressure
```

## Build & Run

```bash
./build.sh
./build/occupancy_tradeoffs
```
