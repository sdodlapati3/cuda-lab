# Week 7 Checkpoint Quiz: Memory Optimization

## Instructions
- Total points: 30
- Time: 30 minutes
- Passing score: 24/30 (80%)

---

## Part 1: Occupancy (10 points)

### Question 1 (3 points)
Calculate the occupancy for this kernel launch on a GPU with:
- Max 2048 threads per SM
- Max 64 warps per SM
- 65536 registers per SM
- Kernel uses 32 registers per thread
- Block size: 256 threads

**Your answer:**

---

### Question 2 (3 points)
What are the three main occupancy limiters? For each, explain how it can limit occupancy.

**Your answer:**

---

### Question 3 (4 points)
A kernel has 25% occupancy due to register pressure. List two strategies to improve occupancy without changing the algorithm.

**Your answer:**

---

## Part 2: Register Optimization (10 points)

### Question 4 (3 points)
What happens when a kernel uses more registers than available per thread? Explain the performance implications.

**Your answer:**

---

### Question 5 (4 points)
Analyze this code for register pressure:

```cpp
__global__ void compute(float* data, int n) {
    float a, b, c, d, e, f, g, h;  // 8 float registers
    float result = 0.0f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    a = data[idx];
    b = data[idx + n];
    c = data[idx + 2*n];
    d = data[idx + 3*n];
    e = a * b + c * d;
    f = a - b * c / d;
    g = e + f;
    h = g * g;
    result = h + e - f;
    
    data[idx] = result;
}
```

How many registers does this likely use? How could you reduce register usage?

**Your answer:**

---

### Question 6 (3 points)
What do `__launch_bounds__` attributes do? Write an example.

**Your answer:**

---

## Part 3: Cache Optimization (10 points)

### Question 7 (3 points)
Compare L1 cache and L2 cache in terms of:
- Location
- Size
- Sharing scope

**Your answer:**

---

### Question 8 (4 points)
For a matrix traverse operation, which access pattern is cache-friendly and why?

Option A:
```cpp
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
        sum += matrix[i * cols + j];  // Row-major access
```

Option B:
```cpp
for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
        sum += matrix[i * cols + j];  // Column-major access
```

**Your answer:**

---

### Question 9 (3 points)
When should you use texture memory instead of global memory? List two scenarios.

**Your answer:**

---

## Bonus Question (2 points)
Explain the difference between `cudaMallocManaged` and `cudaMalloc` + `cudaMemcpy`. When would unified memory hurt performance?

**Your answer:**

---

## Answer Key

### Q1: Occupancy Calculation
- 256 threads = 8 warps per block
- Register usage: 256 × 32 = 8192 registers per block
- Max blocks by registers: 65536 / 8192 = 8 blocks
- Max blocks by threads: 2048 / 256 = 8 blocks
- Active warps: 8 blocks × 8 warps = 64 warps
- Occupancy: 64/64 = 100%

### Q2: Three Limiters
1. Threads: Block size limits concurrent blocks
2. Registers: High register usage limits concurrent threads
3. Shared memory: Large shared memory blocks limit concurrent blocks

### Q3: Strategies
1. Use `__launch_bounds__` to hint compiler
2. Reduce variables, reuse registers
3. Trade computation for registers (recompute vs store)

### Q4: Register Spilling
- Excess registers "spill" to local memory (actually DRAM)
- 100x slower than register access
- Can severely degrade performance

### Q5: Register Analysis
- ~10-15 registers likely (8 variables + temporaries + idx)
- Reduce by: computing in-place, reusing variables, unrolling carefully

### Q6: Launch Bounds
```cpp
__global__ void __launch_bounds__(256, 4) kernel(...) {
    // max 256 threads/block, min 4 blocks/SM
}
```
Helps compiler optimize register allocation.

### Q7: L1 vs L2
| Aspect | L1 | L2 |
|--------|----|----|
| Location | Per SM | Shared across GPU |
| Size | ~128KB | 1.5-40MB |
| Scope | SM-local | All SMs |

### Q8: Row-major Access (Option A)
- Row-major access matches memory layout
- Sequential addresses → coalesced loads
- Cache lines fully utilized

### Q9: Texture Memory
1. 2D spatial locality (image processing)
2. Read-only data with interpolation
3. Boundary handling (clamp/wrap)

### Bonus
- Unified: simpler code, automatic migration
- Explicit: more control, no migration overhead
- Unified hurts when: frequent CPU-GPU switches cause thrashing
