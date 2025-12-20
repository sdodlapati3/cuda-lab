# Week 15 Checkpoint Quiz: Dynamic Parallelism

**Total Points: 30**
**Passing Score: 24/30 (80%)**

---

## Section 1: Conceptual Understanding (10 points)

### Question 1 (2 points)
What compilation flags are required to enable CUDA Dynamic Parallelism?

- A) `-arch=sm_35`
- B) `-rdc=true -lcudadevrt`
- C) `-dynamic`
- D) `-enable-cdp`

### Question 2 (2 points)
Which memory types are visible to both parent and child kernels?

- A) Shared memory only
- B) Local memory only
- C) Global memory and unified memory
- D) All memory types

### Question 3 (2 points)
What is the default maximum nesting depth for dynamic parallelism?

- A) 8 levels
- B) 16 levels
- C) 24 levels
- D) Unlimited

### Question 4 (2 points)
When should you use `cudaDeviceSynchronize()` in a parent kernel?

- A) Never - it's only for host code
- B) Before launching any child kernel
- C) When you need to wait for child kernels to complete
- D) After every line of code

### Question 5 (2 points)
What is "tail launch" optimization in CDP2?

- A) Launching kernels from the last thread
- B) Child kernel reuses parent's resources when parent exits
- C) Launching all children at the end
- D) A special launch syntax

---

## Section 2: Code Analysis (10 points)

### Question 6 (3 points)
What is wrong with this code?

```cuda
__global__ void parent() {
    __shared__ int sharedData[256];
    sharedData[threadIdx.x] = threadIdx.x;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        child<<<1, 256>>>(sharedData);  // Pass shared memory pointer
        cudaDeviceSynchronize();
    }
}
```

**Your answer:** _______________________

### Question 7 (3 points)
This recursive kernel has a bug. Identify and explain it:

```cuda
__global__ void recursive(int* data, int n) {
    if (n > 1) {
        recursive<<<1, 1>>>(data, n / 2);
        recursive<<<1, 1>>>(data + n/2, n / 2);
    }
    // Process data[0..n-1]
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        data[i] *= 2;
    }
}
```

**Your answer:** _______________________

### Question 8 (4 points)
Reorder these steps for correct CDP memory handling:

1. Child kernel reads data
2. Parent calls `cudaDeviceSynchronize()`
3. Parent writes to global memory
4. Parent launches child kernel
5. Parent reads child's results

Correct order: _______________________

---

## Section 3: Practical Application (10 points)

### Question 9 (5 points)
Design a CDP-based algorithm for finding the maximum value in an array recursively. Describe:

1. Base case condition
2. How to divide the problem
3. How to combine results
4. Which memory type to use

**Your design:**
```
Base case: _______________________
Division strategy: _______________________
Combination: _______________________
Memory choice: _______________________
```

### Question 10 (5 points)
Given this scenario, should you use CDP? Explain why or why not.

**Scenario:** You have a regular matrix multiplication of two 1024x1024 matrices where you know the exact dimensions at compile time.

**Your answer:** _______________________

---

## Answer Key

### Section 1
1. **B** - `-rdc=true -lcudadevrt`
2. **C** - Global memory and unified memory
3. **C** - 24 levels (configurable with `cudaDeviceSetLimit`)
4. **C** - When you need to wait for child kernels to complete
5. **B** - Child kernel reuses parent's resources when parent exits

### Section 2
6. **Shared memory cannot be passed to child kernels** - it's not visible across grids. Must use global memory instead.

7. **Missing synchronization before processing** - Parent processes data before children complete. Need `cudaDeviceSynchronize()` after launching children.

8. **Correct order: 3, 4, 2, 1, 5**
   - Parent writes to global memory
   - Parent launches child kernel
   - Parent calls cudaDeviceSynchronize()
   - Child kernel reads data
   - Parent reads child's results

### Section 3
9. **Example design:**
   - Base case: n ≤ 32 (use single-thread max finding)
   - Division: Split array in half, launch two children
   - Combination: Parent compares two child maxes
   - Memory: Unified memory (automatic coherence)

10. **No, CDP is NOT recommended** because:
   - Matrix multiplication has regular, predictable parallelism
   - Dimensions are known at compile time
   - Single kernel launch with tiled approach is more efficient
   - CDP adds overhead without benefit for regular workloads
   - Better to use cuBLAS or a well-optimized single kernel

---

## Scoring Guide

| Score | Assessment |
|-------|------------|
| 28-30 | Excellent - Ready for advanced CDP projects |
| 24-27 | Good - Solid understanding, review weak areas |
| 20-23 | Fair - Review day 1-2 content before proceeding |
| <20 | Needs work - Re-study week 15 materials |
