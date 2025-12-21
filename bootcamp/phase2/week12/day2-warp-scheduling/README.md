# Day 2: Warp Scheduling

## Learning Objectives

- Understand how SMs schedule warps
- Learn about warp states
- See scheduler in action

## Key Concepts

### Warp States

```
┌─────────────┐
│  ELIGIBLE   │ → Can issue instruction
└─────────────┘
       ↓
┌─────────────┐
│  STALLED    │ → Waiting for something
└─────────────┘
```

### Stall Reasons

| Reason | Cause |
|--------|-------|
| Long Scoreboard | Memory load pending |
| Short Scoreboard | Math dependency |
| Memory Throttle | Too many outstanding loads |
| Sync | At __syncthreads() |
| Not Selected | Other warps were chosen |

### Scheduler Operation

```
Each cycle:
  1. Check all warps for eligibility
  2. Select up to N eligible warps (SM dependent)
  3. Issue instructions from selected warps
  4. Update warp states
```

## Build & Run

```bash
./build.sh
./build/warp_scheduling
```
