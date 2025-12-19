# Week 3 Checkpoint Quiz: Parallel Patterns I - Vector Operations

## Instructions
- Answer all questions (30 points total)
- Passing score: 24+ points (80%)
- Review the notebooks if needed before answering

---

## Section 1: Grid-Stride Loops (8 points)

### Q1 (2 pts)
What is the main advantage of grid-stride loops over naive indexing?

```python
# Naive
tid = cuda.grid(1)
if tid < n:
    out[tid] = ...

# Grid-stride
tid = cuda.grid(1)
stride = cuda.gridsize(1)
for i in range(tid, n, stride):
    out[i] = ...
```

A) Grid-stride is faster for small arrays  
B) Grid-stride handles any array size regardless of grid configuration  
C) Grid-stride uses less memory  
D) Grid-stride requires fewer registers  

---

### Q2 (2 pts)
Given `blocks=256, threads=128`, what is `cuda.gridsize(1)`?

A) 256  
B) 128  
C) 32,768  
D) 384  

---

### Q3 (2 pts)
Fill in the blank for a grid-stride loop:

```python
tid = cuda.grid(1)
stride = cuda.gridsize(1)
for i in range(tid, n, _____):
    out[i] = x[i] * 2
```

A) n  
B) tid  
C) stride  
D) 1  

---

### Q4 (2 pts)
For 2D grid-stride loops, how do you get the stride in the y-direction?

```python
x, y = cuda.grid(2)
stride_x = cuda.gridsize(2)[0]
stride_y = _____
```

A) `cuda.gridsize(2)[1]`  
B) `cuda.gridsize(1)`  
C) `cuda.blockDim.y`  
D) `cuda.gridDim.y`  

---

## Section 2: Element-wise Operations (8 points)

### Q5 (2 pts)
Which library provides device math functions in Numba CUDA?

A) numpy  
B) cupy  
C) math (standard library)  
D) cuda.math  

---

### Q6 (2 pts)
What's wrong with this element-wise kernel?

```python
@cuda.jit
def my_kernel(x, out, n):
    tid = cuda.grid(1)
    if tid < n:
        out[tid] = np.sqrt(x[tid])  # Problem here
```

A) Should use `if tid <= n`  
B) Can't use numpy functions in device code  
C) Missing grid-stride loop  
D) Out array should be float64  

---

### Q7 (2 pts)
For GPU-accelerated sigmoid: `σ(x) = 1 / (1 + e^(-x))`, what's the recommended implementation?

A) Use numpy's sigmoid  
B) Call `cuda.sigmoid(x)`  
C) Use `1.0 / (1.0 + math.exp(-x))`  
D) Pre-compute and use lookup table  

---

### Q8 (2 pts)
Which statement is TRUE about element-wise operations on GPUs?

A) They are always compute-bound  
B) They are typically memory-bound  
C) They cannot benefit from parallelism  
D) They require shared memory for good performance  

---

## Section 3: BLAS & SAXPY (6 points)

### Q9 (2 pts)
What does SAXPY stand for and compute?

A) **S**ingle precision **A** times **X** **P**lus **Y**: `y = a*x + y`  
B) **S**um of **A**ll **X** and **Y**: `sum(x) + sum(y)`  
C) **S**calar **A**dd **X** and **Y**: `x + y`  
D) **S**quare **A**rray **X** by **Y**: `x * y`  

---

### Q10 (2 pts)
For SAXPY with n=1 million float32 elements, what is the theoretical memory traffic?

A) 4 MB (read x only)  
B) 8 MB (read x, write y)  
C) 12 MB (read x, read y, write y)  
D) 16 MB (read x, write x, read y, write y)  

---

### Q11 (2 pts)
A GPU with 400 GB/s bandwidth achieves what maximum SAXPY GFLOP/s for large arrays?

*(Hint: SAXPY is 2 FLOPs per element, 12 bytes per element)*

A) ~33 GFLOP/s  
B) ~67 GFLOP/s  
C) ~100 GFLOP/s  
D) ~400 GFLOP/s  

---

## Section 4: Fused Operations (8 points)

### Q12 (2 pts)
What is the PRIMARY benefit of kernel fusion?

A) Reduces code complexity  
B) Reduces memory traffic and kernel launch overhead  
C) Enables use of shared memory  
D) Increases thread utilization  

---

### Q13 (2 pts)
Computing `z = a*x + b*y + c` with separate kernels requires approximately how many memory operations per element?

A) 2 (read x, read y)  
B) 4 (read x, write temp, read y, write temp)  
C) 8 (multiple intermediate reads/writes)  
D) 1 (read only)  

---

### Q14 (2 pts)
When should you NOT fuse kernels?

A) When operations are element-wise  
B) When intermediate results are needed for multiple downstream operations  
C) When reducing memory traffic  
D) When operations use the same input arrays  

---

### Q15 (2 pts)
Which fused operation pattern is SO common that GPUs have hardware support?

A) ReLU  
B) Softmax  
C) Fused Multiply-Add (FMA): `a*b + c`  
D) Convolution  

---

## Bonus Question (2 pts)

### Q16 (2 pts)
You need to compute `z = sqrt((a*x + b)^2 + (c*y + d)^2)` for 10M elements. How many kernels would you use?

A) 1 (fully fused)  
B) 2 (compute each term separately)  
C) 5 (one per operation)  
D) 7 (one per mathematical operator)  

---

## Answer Key

<details>
<summary>Click to reveal answers</summary>

| Q | Answer | Explanation |
|---|--------|-------------|
| 1 | B | Grid-stride handles any n regardless of grid size |
| 2 | C | 256 × 128 = 32,768 total threads |
| 3 | C | Loop increments by stride |
| 4 | A | Second element of gridsize(2) tuple |
| 5 | C | Standard math library works in device code |
| 6 | B | NumPy not available in device code |
| 7 | C | Use math module for device code |
| 8 | B | Simple ops are memory-bound on GPU |
| 9 | A | Single precision A×X Plus Y |
| 10 | C | Read x (4MB) + Read y (4MB) + Write y (4MB) = 12MB |
| 11 | B | (400 GB/s ÷ 12 bytes) × 2 FLOPs ≈ 67 GFLOP/s |
| 12 | B | Main benefit is reduced memory traffic |
| 13 | C | ~8 operations with temporaries |
| 14 | B | Keep separate when intermediate results reused |
| 15 | C | FMA has dedicated hardware instruction |
| 16 | A | Single fused kernel is optimal |

**Scoring:**
- 30-32: Excellent! Ready for Week 4
- 24-29: Good understanding, review weaker areas
- 18-23: Review notebooks before proceeding
- <18: Revisit Week 3 materials

</details>

---

## Self-Reflection

After completing this quiz, consider:

1. **Grid-stride loops:** Can you explain why they're the professional standard?

2. **Memory bandwidth:** Can you calculate theoretical vs actual bandwidth?

3. **Fusion decisions:** Can you identify fusion opportunities in your code?

4. **Performance thinking:** Do you consider memory traffic when designing kernels?

---

## Ready for Week 4?

If you scored 24+, you're ready for **Week 4: Reduction & Atomics**!

Preview:
- Parallel reduction patterns (sum, max, min)
- Warp-level shuffle operations
- Atomic operations for thread-safe updates
- Histogram computation
