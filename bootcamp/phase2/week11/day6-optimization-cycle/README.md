# Day 6: Optimization Cycle

## Learning Objectives

- Implement the profile-optimize-verify loop
- Track optimization progress with metrics
- Know when to stop optimizing

## The Optimization Cycle

```
┌─────────────────────────────────────────────┐
│  1. PROFILE: Identify current bottleneck    │
│                    ↓                        │
│  2. HYPOTHESIZE: What optimization helps?   │
│                    ↓                        │
│  3. IMPLEMENT: Make targeted change         │
│                    ↓                        │
│  4. VERIFY: Re-profile, measure improvement │
│                    ↓                        │
│  5. REPEAT: Until at ceiling or acceptable  │
└─────────────────────────────────────────────┘
```

## Key Principles

### Change One Thing at a Time
- Makes it clear what helped (or hurt)
- Easier to roll back if needed

### Track Metrics Across Iterations
- Time, bandwidth, occupancy
- Compare to theoretical limits

### Know When to Stop
- At roofline ceiling → algorithm change needed
- "Good enough" for your use case
- Diminishing returns

## Build & Run

```bash
./build.sh
./build/optimization_cycle
```
