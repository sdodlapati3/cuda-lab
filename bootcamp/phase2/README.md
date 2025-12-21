# Phase 2: Performance Mental Models

> **Duration:** 4 weeks (Weeks 9-12)
> **Goal:** Measure and optimize with intent. Build the mental models that separate good from great.

## Prerequisites

- ✅ Completed Phase 0 (build, debug, profile infrastructure)
- ✅ Completed Phase 1 (execution model, memory hierarchy, real kernels)
- ✅ Passed [Phase 1 Checkpoint Quiz](../phase1/checkpoint-quiz.md) with >80%

---

## Weekly Schedule

| Week | Topic | Focus |
|------|-------|-------|
| [Week 9](week09/) | Roofline Model | Arithmetic intensity, memory vs compute bound |
| [Week 10](week10/) | Occupancy Deep Dive | Theoretical vs achieved, tradeoffs |
| [Week 11](week11/) | Profiling Mastery | Nsight metrics, bottleneck identification |
| [Week 12](week12/) | Latency Hiding | ILP, MLP, fusion, launch overhead |

---

## Official Documentation Mapping

### CUDA Best Practices Guide Sections
| Topic | Section |
|-------|---------|
| Memory Optimizations | Ch. 5 |
| Execution Configuration | Ch. 10 |
| Instruction Optimization | Ch. 11 |
| Control Flow | Ch. 12 |

### Key Whitepapers
| Resource | Purpose |
|----------|---------|
| [Roofline Model Paper](https://dl.acm.org/doi/10.1145/1498765.1498785) | Original roofline concept |
| [NVIDIA Nsight Compute Docs](https://docs.nvidia.com/nsight-compute/) | Metric definitions |
| [Volta/Ampere Architecture Guides](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Hardware capabilities |

---

## Phase 2 Gate

**You can proceed to Phase 3 when:**

- [ ] Can calculate arithmetic intensity for any kernel
- [ ] Can plot kernels on roofline and interpret position
- [ ] Understand why high occupancy ≠ high performance
- [ ] Can identify memory-bound vs compute-bound from profiler data
- [ ] Have optimization workflow: profile → hypothesis → change → measure
- [ ] All Phase 1 kernels have roofline analysis

---

## Deliverables

1. **Roofline plot** of all Phase 1 kernels with analysis
2. **Optimization case study**: kernel before/after with profiler evidence
3. **Performance report**: documented optimization journey
4. **Launch config experiments**: showing occupancy vs performance tradeoffs

---

## Key Mental Models

### The Roofline Model
```
Performance (FLOPS/s)
     ^
     |          _______________  Peak Compute
     |         /
     |        /
     |       /
     |      /   <- Memory-bound region
     |     /
     |    /
     +---+-------------------------> Arithmetic Intensity (FLOPS/Byte)
        ^
        Peak Bandwidth slope
```

**Key insight:** Most kernels are memory-bound. Optimize memory first.

### Occupancy vs Performance
```
High Occupancy ≠ High Performance

Low occupancy CAN be fast if:
- Enough ILP hides latency
- Registers are well-utilized
- Memory access is efficient

High occupancy CAN be slow if:
- Register spilling occurs
- Bank conflicts in shared memory
- Poor memory access patterns
```

### The Optimization Cycle
```
┌─────────────────────────────────────┐
│  1. PROFILE                         │
│     - Identify bottleneck           │
│     - Memory or compute bound?      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. HYPOTHESIZE                     │
│     - What limits performance?      │
│     - What change would help?       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. IMPLEMENT                       │
│     - Make ONE change               │
│     - Keep old version for compare  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. MEASURE                         │
│     - Did performance improve?      │
│     - Validate hypothesis           │
└──────────────┬──────────────────────┘
               │
               ▼
        (Repeat until satisfied)
```

---

## Week Overview

### Week 9: Roofline Model
Build the fundamental performance model every GPU developer needs.

- **Day 1:** Arithmetic intensity concept
- **Day 2:** Memory bandwidth measurement
- **Day 3:** Compute throughput measurement
- **Day 4:** Building roofline plots
- **Day 5:** Analyzing kernel positions
- **Day 6:** Optimization strategies by position

### Week 10: Occupancy Deep Dive
Understand the nuanced relationship between occupancy and performance.

- **Day 1:** Occupancy calculation
- **Day 2:** Register pressure effects
- **Day 3:** Shared memory tradeoffs
- **Day 4:** Launch configuration experiments
- **Day 5:** When low occupancy wins
- **Day 6:** Occupancy calculator tools

### Week 11: Profiling Mastery
Master Nsight Compute and Nsight Systems for production optimization.

- **Day 1:** Key NCU metrics
- **Day 2:** Memory throughput analysis
- **Day 3:** Compute throughput analysis
- **Day 4:** Stall analysis
- **Day 5:** Nsight Systems for multi-kernel workflows
- **Day 6:** Profiling-driven optimization workflow

### Week 12: Latency Hiding
Understand how GPUs hide latency and how to help them.

- **Day 1:** Instruction-level parallelism (ILP)
- **Day 2:** Memory-level parallelism (MLP)
- **Day 3:** Kernel launch overhead
- **Day 4:** Kernel fusion principles
- **Day 5:** When fusion matters
- **Day 6:** Putting it all together

---

## Tools Mastery Checklist

### Nsight Compute (ncu)
- [ ] Basic profiling: `ncu ./app`
- [ ] Save reports: `ncu -o report ./app`
- [ ] Specific metrics: `ncu --metrics <metric_list> ./app`
- [ ] Kernel selection: `ncu --kernel-name <name> ./app`
- [ ] Compare reports: `ncu --compare`

### Nsight Systems (nsys)
- [ ] Timeline capture: `nsys profile ./app`
- [ ] Analyze concurrency
- [ ] Identify gaps/stalls
- [ ] Multi-GPU analysis

### Roofline Analysis
- [ ] Measure peak bandwidth
- [ ] Measure peak compute
- [ ] Calculate arithmetic intensity
- [ ] Plot and interpret

---

## Common Pitfalls

1. **Optimizing without profiling** → Wasted effort on non-bottlenecks
2. **Assuming high occupancy = fast** → Missing ILP/register opportunities
3. **Ignoring memory bandwidth** → Most kernels are memory-bound
4. **Making multiple changes at once** → Can't attribute improvements
5. **Not measuring variance** → Misleading performance conclusions
