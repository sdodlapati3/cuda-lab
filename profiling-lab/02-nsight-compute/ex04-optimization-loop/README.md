# Exercise 04: The Optimization Loop

## Learning Objectives
- Practice iterative profile-optimize-verify workflow
- Build intuition for interpreting metrics
- Document optimization decisions
- Achieve measurable performance improvements

## The Optimization Loop

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   1. PROFILE                                                │
│   ────────────                                              │
│   - Establish baseline metrics                              │
│   - Identify primary bottleneck                             │
│                                                             │
│              ↓                                              │
│                                                             │
│   2. HYPOTHESIZE                                            │
│   ─────────────                                             │
│   - What optimization will address this bottleneck?         │
│   - What metric should improve?                             │
│   - What's the expected speedup?                            │
│                                                             │
│              ↓                                              │
│                                                             │
│   3. IMPLEMENT                                              │
│   ────────────                                              │
│   - Make ONE change at a time                               │
│   - Keep code readable                                      │
│                                                             │
│              ↓                                              │
│                                                             │
│   4. VERIFY                                                 │
│   ────────                                                  │
│   - Correctness first!                                      │
│   - Measure new metrics                                     │
│   - Did hypothesis hold?                                    │
│                                                             │
│              ↓                                              │
│                                                             │
│   5. DOCUMENT                                               │
│   ──────────                                                │
│   - Record before/after metrics                             │
│   - Explain why it worked (or didn't)                       │
│   - Update roofline position                                │
│                                                             │
│              ↓                                              │
│                                                             │
│   Repeat until: close to peak OR diminishing returns        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Exercise: Optimize a Matrix Transpose

Starting point: Naive matrix transpose (terrible performance)

### Iteration 0: Baseline

```cpp
// naive_transpose.cu
__global__ void naive_transpose(float* out, float* in, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < N && y < N) {
        out[y * N + x] = in[x * N + y];  // Uncoalesced writes!
    }
}
```

**Profile:**
```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    -o iteration0 ./naive_transpose
```

**Record baseline:**
| Metric | Value |
|--------|-------|
| Time | 2.5 ms |
| Memory Throughput | 150 GB/s (15% of peak) |
| Global Store Efficiency | 12.5% |

**Bottleneck identified:** Uncoalesced global memory writes

---

### Iteration 1: Coalesced Reads

**Hypothesis:** Swap loop order to coalesce reads instead of writes

**Implementation:**
```cpp
__global__ void coalesced_read_transpose(float* out, float* in, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < N && y < N) {
        out[x * N + y] = in[y * N + x];  // Coalesced reads, uncoalesced writes
    }
}
```

**Expected:** ~2x speedup (reads more common than writes in cache)

**Profile:**
```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    -o iteration1 ./coalesced_read_transpose
```

**Record:**
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Time | 2.5 ms | 1.8 ms | -28% |
| Memory Throughput | 150 GB/s | 210 GB/s | +40% |
| Global Load Efficiency | 12.5% | 100% | ✓ |
| Global Store Efficiency | 100% | 12.5% | Still bad |

**Analysis:** Helped but writes still uncoalesced. Need shared memory.

---

### Iteration 2: Shared Memory Tile

**Hypothesis:** Use shared memory to coalesce both reads and writes

**Implementation:**
```cpp
#define TILE_DIM 32

__global__ void shared_mem_transpose(float* out, float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Coalesced read into shared memory
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    
    __syncthreads();
    
    // Transposed indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    // Coalesced write from shared memory
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}
```

**Expected:** ~5-10x over baseline

**Profile:**
```bash
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    -o iteration2 ./shared_mem_transpose
```

**Record:**
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Time | 1.8 ms | 0.4 ms | -78% |
| Memory Throughput | 210 GB/s | 800 GB/s | +280% |
| Global Load Efficiency | 100% | 100% | ✓ |
| Global Store Efficiency | 12.5% | 100% | ✓ Fixed! |

**Analysis:** Major improvement! But check for bank conflicts...

---

### Iteration 3: Avoid Bank Conflicts

**Hypothesis:** Shared memory bank conflicts are limiting throughput

**Implementation:**
```cpp
#define TILE_DIM 32
#define PADDING 1

__global__ void bank_conflict_free_transpose(float* out, float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + PADDING];  // +1 padding
    // ... rest same as before
}
```

**Profile:**
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared \
    -o iteration3 ./bank_conflict_free_transpose
```

**Record:**
| Metric | Before | After |
|--------|--------|-------|
| Bank conflicts | 1024 | 0 |
| Time | 0.4 ms | 0.35 ms |
| Memory Throughput | 800 GB/s | 900 GB/s |

---

## Your Task

Apply this methodology to optimize one of:

1. **Reduction kernel** - Start with naive, achieve >80% of peak bandwidth
2. **1D Convolution** - Use shared memory for halo regions
3. **Histogram** - Handle atomic contention

## Documentation Template

For each iteration, record:

```markdown
## Iteration N: [Optimization Name]

### Hypothesis
- Bottleneck: [what metric shows the problem]
- Optimization: [what you'll change]
- Expected improvement: [% or metric target]

### Implementation
[Code changes or description]

### Results
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Time | | | |
| Throughput | | | |
| Key metric | | | |

### Analysis
[Why did it work? What's the new bottleneck?]
```

## Success Criteria

- [ ] Completed at least 3 optimization iterations
- [ ] Each iteration has before/after metrics
- [ ] Final version achieves >70% of theoretical peak
- [ ] Can explain why each optimization helped
- [ ] Documented the optimization journey
