# Week 2 Checkpoint Quiz: Memory Patterns & Optimization

**Total Points: 30**  
**Time: 20-30 minutes**  
**Passing Score: 24/30 (80%)**

---

## Section A: Memory Coalescing (8 points)

### Question 1 (2 points)
What does "memory coalescing" mean in CUDA?

- [ ] a) Combining multiple GPU devices into one
- [ ] b) Merging multiple memory requests from threads in a warp into fewer transactions
- [ ] c) Copying memory from CPU to GPU
- [ ] d) Freeing unused memory automatically

<details>
<summary>Answer</summary>
b) Merging multiple memory requests from threads in a warp into fewer transactions
</details>

### Question 2 (2 points)
Consider this memory access pattern where `tid = threadIdx.x`:
```python
data[tid * 2]  # Each thread accesses every other element
```

This access pattern is:
- [ ] a) Fully coalesced (1 transaction per 32 threads)
- [ ] b) Partially coalesced (2 transactions per 32 threads)
- [ ] c) Not coalesced (32 transactions per 32 threads)
- [ ] d) Invalid (will cause an error)

<details>
<summary>Answer</summary>
b) Partially coalesced - stride-2 access means threads access every other cache line, requiring 2 transactions instead of 1.
</details>

### Question 3 (2 points)
For a 2D array stored in row-major order, which access pattern is coalesced?

- [ ] a) Adjacent threads access adjacent rows: `data[tid, 0]`
- [ ] b) Adjacent threads access adjacent columns: `data[0, tid]`
- [ ] c) Both are equally coalesced
- [ ] d) Neither is coalesced

<details>
<summary>Answer</summary>
b) Adjacent threads accessing adjacent columns: `data[0, tid]` - this accesses contiguous memory in row-major layout.
</details>

### Question 4 (2 points)
A cache line on modern NVIDIA GPUs is:
- [ ] a) 32 bytes
- [ ] b) 64 bytes
- [ ] c) 128 bytes
- [ ] d) 256 bytes

<details>
<summary>Answer</summary>
c) 128 bytes - this is why 32 threads accessing 32-bit (4-byte) floats results in exactly one 128-byte transaction when coalesced.
</details>

---

## Section B: Shared Memory (8 points)

### Question 5 (2 points)
What is the primary advantage of shared memory over global memory?

- [ ] a) Larger capacity
- [ ] b) Accessible by all threads in the grid
- [ ] c) On-chip location with ~100x lower latency
- [ ] d) Automatic persistence between kernel launches

<details>
<summary>Answer</summary>
c) On-chip location with ~100x lower latency
</details>

### Question 6 (2 points)
Shared memory is shared among:

- [ ] a) All threads in the grid
- [ ] b) All threads in a block
- [ ] c) All threads in a warp
- [ ] d) Only a single thread

<details>
<summary>Answer</summary>
b) All threads in a block
</details>

### Question 7 (2 points)
Why is `cuda.syncthreads()` necessary after loading data into shared memory?

- [ ] a) To free up registers
- [ ] b) To ensure all threads have finished loading before any thread reads
- [ ] c) To copy shared memory back to global memory
- [ ] d) To clear the shared memory for reuse

<details>
<summary>Answer</summary>
b) To ensure all threads have finished loading before any thread reads - prevents race conditions.
</details>

### Question 8 (2 points)
In a tiled matrix multiplication, using shared memory reduces global memory traffic by:

- [ ] a) Storing the output matrix in shared memory
- [ ] b) Allowing each element of A and B to be loaded once per tile instead of once per output element
- [ ] c) Compressing the matrix data
- [ ] d) Using lower precision arithmetic

<details>
<summary>Answer</summary>
b) Allowing each element of A and B to be loaded once per tile instead of once per output element - data reuse across threads.
</details>

---

## Section C: Bank Conflicts (8 points)

### Question 9 (2 points)
How many banks does shared memory have on modern NVIDIA GPUs?

- [ ] a) 16
- [ ] b) 32
- [ ] c) 64
- [ ] d) 128

<details>
<summary>Answer</summary>
b) 32 banks - each bank can service one 32-bit word per cycle.
</details>

### Question 10 (2 points)
A 32-way bank conflict means:

- [ ] a) Memory access is 32x slower
- [ ] b) All 32 threads access the same bank, causing serial access
- [ ] c) 32 different banks are accessed simultaneously
- [ ] d) The kernel will fail to launch

<details>
<summary>Answer</summary>
b) All 32 threads access the same bank, causing serial access - worst-case serialization.
</details>

### Question 11 (2 points)
Which access pattern causes NO bank conflicts?

- [ ] a) `shared[threadIdx.x]` (stride 1)
- [ ] b) `shared[threadIdx.x * 32]` (stride 32)
- [ ] c) `shared[threadIdx.x * 16]` (stride 16)
- [ ] d) Both a and b

<details>
<summary>Answer</summary>
d) Both a and b - Stride 1 accesses consecutive banks; stride 32 causes all threads to access the same bank, BUT if they access the SAME address, it's a broadcast (free). Different addresses at stride 32 would cause conflicts.

Actually, the correct answer is a). Stride 1 accesses consecutive banks with no conflicts. Stride 32 causes all threads to hit the same bank (conflict unless same address).
</details>

### Question 12 (2 points)
The "padding trick" to avoid bank conflicts works by:

- [ ] a) Adding extra elements to shift column access patterns across different banks
- [ ] b) Reducing the size of shared memory
- [ ] c) Using atomic operations
- [ ] d) Increasing thread block size

<details>
<summary>Answer</summary>
a) Adding extra elements to shift column access patterns across different banks - e.g., `[N][N+1]` instead of `[N][N]` for power-of-2 N.
</details>

---

## Section D: Special Memory Types (6 points)

### Question 13 (2 points)
Constant memory is ideal for:

- [ ] a) Large arrays that change every kernel launch
- [ ] b) Small, read-only data where all threads read the same values
- [ ] c) Thread-private temporary variables
- [ ] d) Block-wide communication

<details>
<summary>Answer</summary>
b) Small, read-only data where all threads read the same values - benefits from broadcast capability.
</details>

### Question 14 (2 points)
What is the maximum size of constant memory per GPU?

- [ ] a) 16 KB
- [ ] b) 48 KB
- [ ] c) 64 KB
- [ ] d) 128 KB

<details>
<summary>Answer</summary>
c) 64 KB
</details>

### Question 15 (2 points)
Texture memory is optimized for:

- [ ] a) Sequential 1D access patterns
- [ ] b) 2D spatial locality and hardware interpolation
- [ ] c) Atomic read-modify-write operations
- [ ] d) Write-heavy workloads

<details>
<summary>Answer</summary>
b) 2D spatial locality and hardware interpolation - the texture cache is optimized for 2D/3D access patterns.
</details>

---

## Practical Exercise (Bonus - 5 points)

Given this memory access pattern, identify ALL the issues:

```python
@cuda.jit
def problematic_kernel(input_2d, output):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    # Shared memory
    shared = cuda.shared.array((32, 32), dtype=float32)
    
    # Load - Issue #1?
    shared[tx, ty] = input_2d[tx, ty]
    
    # Process - Issue #2?
    result = shared[ty, tx]  # Transposed access
    
    # Store - Issue #3?
    output[tx * 32 + ty] = result
```

<details>
<summary>Answer</summary>

**Issue #1: Non-coalesced global load**
- `input_2d[tx, ty]` - threads with adjacent `tx` values access adjacent rows, not columns
- Fix: `input_2d[ty, tx]` or swap tx/ty indices

**Issue #2: Missing syncthreads + Bank conflicts**
- No `cuda.syncthreads()` between load and read
- `shared[ty, tx]` when tx varies across threads causes stride-32 access (bank conflicts)
- Fix: Add syncthreads and use padding `shared[32, 33]`

**Issue #3: Non-coalesced global store**
- `output[tx * 32 + ty]` - stride-32 access pattern is not coalesced
- Fix: `output[ty * 32 + tx]` for coalesced writes

</details>

---

## Self-Assessment

Calculate your score and check your readiness:

| Score | Assessment |
|-------|------------|
| 28-30 | Excellent! Ready for Week 3 |
| 24-27 | Good understanding, review missed concepts |
| 20-23 | Review Days 1-4 before proceeding |
| Below 20 | Strongly recommend re-doing Week 2 exercises |

---

## Quick Reference Card

### Memory Coalescing Rules
```
✓ Adjacent threads → adjacent memory addresses
✓ 32 threads × 4-byte access = 128-byte cache line
✓ Stride-1 access = 1 transaction (ideal)
✓ Stride-N access = more transactions (N > 1)
```

### Shared Memory Best Practices
```
✓ Use for data reused by multiple threads
✓ Always syncthreads() after loading
✓ Size limited per SM (48-164 KB)
✓ Use padding to avoid bank conflicts
```

### Bank Conflict Avoidance
```
✓ 32 banks, 4-byte words per bank
✓ Stride-1: No conflict
✓ Stride-2,4,8,16: Conflicts
✓ Stride-32: All same bank (worst)
✓ Padding: array[N][N+1] instead of [N][N]
```

### Memory Selection Guide
```
Small read-only, uniform access → Constant
2D spatial, interpolation      → Texture  
Block-local reuse             → Shared
Thread-private temp           → Register
Large arrays, streaming       → Global
```
