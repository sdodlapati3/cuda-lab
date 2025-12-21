# Day 6: Putting It All Together - Phase 2 Capstone

## Learning Objectives

- Integrate all mental models from Phase 2
- Apply systematic optimization workflow
- Build your optimization checklist

## The Complete Mental Model

```
                    PERFORMANCE ANALYSIS FRAMEWORK
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    STEP 1: ROOFLINE                         │
    │  Is your kernel memory-bound or compute-bound?              │
    │  → Calculate Arithmetic Intensity (FLOPs / Bytes)           │
    │  → Compare to hardware ridgeline                            │
    └─────────────────────────────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                   STEP 2: OCCUPANCY                         │
    │  Are you using the SM efficiently?                          │
    │  → Check registers per thread                               │
    │  → Check shared memory per block                            │
    │  → Find optimal block size                                  │
    └─────────────────────────────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                   STEP 3: PROFILING                         │
    │  What does the hardware say?                                │
    │  → nsys for timeline overview                               │
    │  → ncu for detailed kernel analysis                         │
    │  → Identify the actual bottleneck                           │
    └─────────────────────────────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                STEP 4: LATENCY HIDING                       │
    │  Are you hiding latencies?                                  │
    │  → Memory: prefetch, async copy, double buffer              │
    │  → Compute: ILP, multiple accumulators, unrolling           │
    │  → TLP: enough warps to hide stalls                         │
    └─────────────────────────────────────────────────────────────┘
```

## Optimization Checklist

### Memory-Bound Kernels
- [ ] Coalesced global memory access?
- [ ] Using shared memory for reuse?
- [ ] Avoiding bank conflicts?
- [ ] Vectorized loads (float4)?
- [ ] Async copies (Ampere+)?

### Compute-Bound Kernels
- [ ] Minimized register pressure?
- [ ] Using fast math where possible?
- [ ] Loop unrolling?
- [ ] Multiple accumulators?
- [ ] Tensor cores if applicable?

### General
- [ ] Optimal block size for occupancy?
- [ ] Enough parallelism for GPU?
- [ ] Host-device transfers minimized?
- [ ] Streams for overlap?

## Build & Run

```bash
./build.sh
./build/integration_demo
```
