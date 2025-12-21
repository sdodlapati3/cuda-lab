# Day 4: Warp Execution & Divergence

## Learning Objectives

- Understand SIMT (Single Instruction, Multiple Threads)
- Recognize warp divergence and its performance impact
- Write divergence-free code patterns

## Key Concepts

### Warp Execution Model

- A **warp** is 32 threads executing in lockstep
- All threads in a warp execute the **same instruction**
- This is called SIMT: Single Instruction, Multiple Threads
- Hardware schedules at warp granularity, not thread

### Warp Divergence

When threads in a warp take different paths (if/else), both paths must execute:

```cuda
if (threadIdx.x < 16) {
    // Path A: Only first 16 threads need this
    expensive_operation_A();
} else {
    // Path B: Only last 16 threads need this
    expensive_operation_B();
}
// Both paths execute serially - 2× time!
```

### Divergence Patterns

| Pattern | Divergence | Why |
|---------|------------|-----|
| `if (idx < N)` | Minimal | Only last warp affected |
| `if (threadIdx.x < 16)` | Yes | Half warp inactive |
| `if (threadIdx.x % 2 == 0)` | Yes | Alternating threads |
| `if (warp_id % 2 == 0)` | None | Whole warps take same path |

### Good Practice

Structure code so entire warps take the same path:
```cuda
int warp_id = threadIdx.x / 32;
if (warp_id < 2) {
    // First 2 warps do this - no divergence!
}
```

## Exercises

1. **Measure Divergence**: Compare divergent vs non-divergent kernels
2. **Profile with Nsight**: See stalled cycles from divergence
3. **Optimize Pattern**: Convert divergent code to warp-uniform

## Build & Run

```bash
./build.sh
./build/warp_divergence
./build/simt_demo
```
