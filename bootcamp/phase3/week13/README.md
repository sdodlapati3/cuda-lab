# Week 13: Warp-Level Programming

## Theme: Mastering the Fundamental Unit of GPU Execution

The warp (32 threads) is the true unit of execution on NVIDIA GPUs. This week, you'll learn to exploit warp-level operations for maximum efficiency.

## Daily Breakdown

| Day | Topic | Focus |
|-----|-------|-------|
| 1 | Warp Fundamentals | SIMT model, warp formation, divergence |
| 2 | Shuffle Instructions | Direct thread-to-thread communication |
| 3 | Warp Reductions | Sum, min, max without shared memory |
| 4 | Warp Scans | Prefix sums within a warp |
| 5 | Vote Functions | Collective boolean operations |
| 6 | Warp-Level Patterns | Building blocks for algorithms |

## Key Insight

```
Traditional (with shared memory):
  Thread 0 → smem[0] → sync → Thread 1 reads smem[0]
  Cost: ~30 cycles + sync overhead

Warp shuffle:
  Thread 0 → __shfl_sync → Thread 1
  Cost: ~2 cycles, NO sync needed!
```

## Mental Model: The Warp as a Unit

```
┌─────────────────────────────────────────────────────────────┐
│                        WARP (32 threads)                     │
├────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┤
│ T0 │ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │... │T28 │T29 │T30 │T31│
├────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┤
│  All execute SAME instruction at SAME time (SIMT)          │
│  Can communicate via shuffle without shared memory          │
│  Divergence = serialization (avoid!)                        │
└─────────────────────────────────────────────────────────────┘
```

## Shuffle Operations Overview

| Function | Description | Use Case |
|----------|-------------|----------|
| `__shfl_sync` | Get value from specific lane | Broadcast |
| `__shfl_up_sync` | Get value from lane - delta | Prefix scan |
| `__shfl_down_sync` | Get value from lane + delta | Reduction |
| `__shfl_xor_sync` | Get value from lane XOR mask | Butterfly pattern |

## This Week's Goal

By the end of Week 13, you'll be able to:
- Write efficient warp-level reductions
- Use shuffles instead of shared memory where possible
- Implement prefix scans within a warp
- Recognize patterns suited for warp-level operations
