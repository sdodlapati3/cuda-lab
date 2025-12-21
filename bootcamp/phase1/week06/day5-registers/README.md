# Day 5: Registers & Local Memory

## Learning Objectives

- Understand register allocation and limits
- Recognize register spilling
- Manage register pressure for better occupancy

## Key Concepts

### Registers

- **Fastest** memory on GPU (~1 cycle)
- Private to each thread
- Limited per thread (~255 max)
- Limited per SM (65536 typically)

### Register Allocation

Compiler allocates registers based on:
- Local variables
- Intermediate computations
- Array indexing (if small and predictable)

Check with: `nvcc --ptxas-options=-v kernel.cu`

### Register Pressure

High register usage → fewer concurrent threads:
```
Registers/thread × Threads/SM ≤ Registers/SM

Example: 64 registers × 1024 threads = 65536 registers ✓
Example: 128 registers × 1024 threads = 131072 registers ✗ (too many!)
        → Occupancy reduced to 512 threads
```

### Local Memory

When registers spill (too many variables):
- Data goes to **local memory** (in global memory!)
- Per-thread but very slow
- Avoid with: fewer variables, smaller arrays

### __launch_bounds__

Tell compiler your intended launch config:
```cuda
__global__ void __launch_bounds__(256, 4)  // 256 threads, 4 blocks/SM
my_kernel() {
    // Compiler optimizes register usage for this config
}
```

### Reducing Register Pressure

1. **Reuse variables**: Don't keep all intermediates
2. **Avoid large local arrays**: Use shared memory instead
3. **Use #pragma unroll carefully**: Unrolling increases registers
4. **Split kernels**: Separate compute-heavy and memory-heavy phases

## Exercises

1. **View register usage**: Compile with -v and analyze
2. **Cause spilling**: Create kernel that spills and measure impact
3. **Apply launch_bounds**: Optimize for specific configuration

## Build & Run

```bash
./build.sh
./build/register_demo
./build/spill_demo
```
