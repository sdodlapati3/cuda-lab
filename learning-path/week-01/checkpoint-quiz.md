# Week 1 Checkpoint Quiz

Test your understanding before moving to Week 2. Answer without looking at notes first!

---

## Section A: Conceptual Understanding (10 points)

### Q1: CPU vs GPU (2 points)
Which statement is TRUE about GPUs compared to CPUs?

- [ ] A) GPUs have fewer cores but each core is much faster
- [ ] B) GPUs have thousands of simpler cores optimized for throughput
- [ ] C) GPUs are always faster than CPUs for any task
- [ ] D) GPUs have larger per-core caches than CPUs

<details>
<summary>Click for answer</summary>

**B) GPUs have thousands of simpler cores optimized for throughput**

GPUs trade single-thread performance for massive parallelism. They're optimized for throughput (doing many things at once), while CPUs are optimized for latency (doing one thing as fast as possible).
</details>

---

### Q2: When to Use GPU (2 points)
Which task is BEST suited for GPU acceleration?

- [ ] A) Parsing a text file line by line
- [ ] B) Running a decision tree with many branches
- [ ] C) Applying the same filter to every pixel in an image
- [ ] D) Accessing a database with complex queries

<details>
<summary>Click for answer</summary>

**C) Applying the same filter to every pixel in an image**

Image processing is ideal for GPU because:
- Same operation on millions of pixels (SIMD pattern)
- Minimal branching
- Data parallel, independent operations
</details>

---

### Q3: Thread Hierarchy (2 points)
In CUDA, threads are organized into _____, which are organized into _____.

- [ ] A) Grids, Blocks
- [ ] B) Blocks, Grids
- [ ] C) Warps, Threads
- [ ] D) Cores, SMs

<details>
<summary>Click for answer</summary>

**B) Blocks, Grids**

Thread hierarchy (smallest to largest):
- Thread → Block → Grid
- Threads within a block can cooperate (shared memory, sync)
- Blocks within a grid are independent
</details>

---

### Q4: Warp Size (2 points)
What is the warp size in NVIDIA GPUs, and why does it matter?

<details>
<summary>Click for answer</summary>

**Warp size is 32 threads.**

It matters because:
- Threads in a warp execute in lockstep (SIMT)
- Branch divergence within a warp hurts performance
- Memory access patterns should be aligned to warp boundaries
- Warp-level primitives (shuffle, vote) operate on 32 threads
</details>

---

### Q5: Memory Transfer (2 points)
Why is memory transfer between CPU and GPU often the bottleneck?

<details>
<summary>Click for answer</summary>

**PCIe bus bandwidth is much lower than GPU memory bandwidth.**

- PCIe 4.0 x16: ~32 GB/s
- GPU HBM2/GDDR6: 200-900 GB/s

This means:
- Transfer time can exceed compute time for simple operations
- Keep data on GPU as long as possible
- Batch operations before copying back
</details>

---

## Section B: Practical Knowledge (10 points)

### Q6: Global Index Formula (2 points)
Write the formula to calculate a 1D global thread index.

```
global_idx = _______________________________________
```

<details>
<summary>Click for answer</summary>

```
global_idx = blockIdx.x * blockDim.x + threadIdx.x
```

Or in Numba: `cuda.grid(1)`
</details>

---

### Q7: Launch Configuration (2 points)
You have an array of 1,000,000 elements and want 256 threads per block.
How many blocks do you need?

```
blocks = _______________________________________
```

<details>
<summary>Click for answer</summary>

```
blocks = ceil(1,000,000 / 256) = 3,907
```

Always round UP to ensure all elements are covered.
In Python: `math.ceil(1_000_000 / 256)`
</details>

---

### Q8: Boundary Check (2 points)
What's wrong with this kernel? How would you fix it?

```python
@cuda.jit
def broken_kernel(arr):
    idx = cuda.grid(1)
    arr[idx] = idx * 2  # 💥 Bug here
```

<details>
<summary>Click for answer</summary>

**Missing boundary check!** When grid size > array size, threads will access out-of-bounds memory.

**Fixed:**
```python
@cuda.jit
def fixed_kernel(arr, n):
    idx = cuda.grid(1)
    if idx < n:  # Boundary check
        arr[idx] = idx * 2
```
</details>

---

### Q9: Memory Functions (2 points)
Match the Numba function to its purpose:

| Function | Purpose |
|----------|---------|
| `cuda.to_device(arr)` | ___ |
| `cuda.device_array(n, dtype)` | ___ |
| `device_arr.copy_to_host()` | ___ |
| `cuda.synchronize()` | ___ |

A) Allocate on GPU without copying
B) Copy host array to device
C) Wait for all GPU operations to complete
D) Copy device array back to host

<details>
<summary>Click for answer</summary>

| Function | Purpose |
|----------|---------|
| `cuda.to_device(arr)` | **B) Copy host array to device** |
| `cuda.device_array(n, dtype)` | **A) Allocate on GPU without copying** |
| `device_arr.copy_to_host()` | **D) Copy device array back to host** |
| `cuda.synchronize()` | **C) Wait for all GPU operations to complete** |
</details>

---

### Q10: Data Types (2 points)
Why should you prefer `np.float32` over `np.float64` for GPU computing?

<details>
<summary>Click for answer</summary>

**float32 is 2x faster than float64 on most GPUs because:**

1. Half the memory bandwidth required
2. More float32 cores than float64 cores on most GPUs
3. Twice as many elements fit in cache/registers
4. Sufficient precision for most applications

Use float64 only when you specifically need the extra precision.
</details>

---

## Section C: Code Challenge (10 points)

### Q11: Complete the Kernel (5 points)

Complete this 2D image brightness adjustment kernel:

```python
@cuda.jit
def adjust_brightness(image, output, height, width, factor):
    """
    Multiply each pixel by factor to adjust brightness.
    Clamp values to 0-255 range.
    """
    # TODO: Get 2D coordinates
    col, row = ___________________
    
    # TODO: Check boundaries
    if ___________________:
        # TODO: Adjust brightness and clamp
        val = ___________________
        output[row, col] = ___________________
```

<details>
<summary>Click for answer</summary>

```python
@cuda.jit
def adjust_brightness(image, output, height, width, factor):
    col, row = cuda.grid(2)
    
    if row < height and col < width:
        val = image[row, col] * factor
        output[row, col] = min(255.0, max(0.0, val))
```
</details>

---

### Q12: Grid-Stride Loop (5 points)

Rewrite this kernel to use a grid-stride loop so it can handle arrays of ANY size:

```python
@cuda.jit
def double_values(arr, n):
    idx = cuda.grid(1)
    if idx < n:
        arr[idx] *= 2
```

<details>
<summary>Click for answer</summary>

```python
@cuda.jit
def double_values_strided(arr, n):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    while idx < n:
        arr[idx] *= 2
        idx += stride
```

This allows handling arrays larger than the grid size!
</details>

---

## Scoring

| Section | Points | Your Score |
|---------|--------|------------|
| A: Conceptual | 10 | ___ |
| B: Practical | 10 | ___ |
| C: Code | 10 | ___ |
| **Total** | **30** | ___ |

### Interpretation

- **25-30**: Excellent! Ready for Week 2
- **20-24**: Good, but review weak areas
- **15-19**: Need more practice - redo notebooks
- **< 15**: Not ready - restart Week 1

---

## Next Steps

- [ ] Score yourself honestly
- [ ] Review any topics where you struggled
- [ ] Complete any remaining exercises
- [ ] Move on to Week 2 when ready!

---

[← Back to Week 1](./README.md) | [Week 2 →](../week-02/README.md)
