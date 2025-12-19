# Week 4 Checkpoint Quiz: Reduction & Atomics

## Instructions
- Answer all questions (30 points total)
- Passing score: 24+ points (80%)
- Review the notebooks if needed before answering

---

## Section 1: Parallel Reduction (8 points)

### Q1 (2 pts)
What is the time complexity of tree-based parallel reduction for N elements?

A) O(N)  
B) O(log N)  
C) O(N log N)  
D) O(N²)  

---

### Q2 (2 pts)
In a block reduction kernel, why do we use `cuda.syncthreads()` after each reduction step?

A) To save power  
B) To ensure all threads have completed their additions before the next step  
C) To prevent memory overflow  
D) To switch between shared and global memory  

---

### Q3 (2 pts)
For a block of 256 threads, how many reduction steps are needed to get a single value?

A) 256  
B) 128  
C) 8 (log₂256)  
D) 16  

---

### Q4 (2 pts)
In a two-pass reduction, what does the first pass produce?

A) The final answer  
B) One partial result per thread  
C) One partial result per block  
D) A sorted array  

---

## Section 2: Warp Primitives (8 points)

### Q5 (2 pts)
How many threads are in a warp?

A) 16  
B) 32  
C) 64  
D) 128  

---

### Q6 (2 pts)
What does `cuda.shfl_down_sync(0xFFFFFFFF, val, 4)` do?

A) Each thread reads from the thread 4 lanes above  
B) Each thread reads from the thread 4 lanes below  
C) Each thread shifts its value down by 4 bits  
D) Only thread 4 gets the value  

---

### Q7 (2 pts)
Why is warp-level reduction faster than shared memory reduction?

A) Warps have more memory  
B) No synchronization needed within a warp  
C) Warps use faster instructions  
D) Warps have dedicated cache  

---

### Q8 (2 pts)
In warp reduction, how many `shfl_down_sync` operations are needed to reduce 32 values to 1?

A) 32  
B) 16  
C) 5 (log₂32)  
D) 1  

---

## Section 3: Atomic Operations (8 points)

### Q9 (2 pts)
What problem do atomic operations solve?

A) Memory allocation  
B) Race conditions in concurrent updates  
C) Data type conversion  
D) Array indexing  

---

### Q10 (2 pts)
What does `cuda.atomic.add(arr, 0, 5)` return?

A) The new value (old + 5)  
B) The old value before addition  
C) True if successful  
D) The index 0  

---

### Q11 (2 pts)
Why is naive atomic sum (one atomic per element) slow?

A) Atomics use more memory  
B) High contention - threads serialize at the same location  
C) Atomics require more registers  
D) Atomic instructions are inherently slow  

---

### Q12 (2 pts)
Which memory type has faster atomic operations?

A) Global memory  
B) Shared memory  
C) Constant memory  
D) Texture memory  

---

## Section 4: Histogram (6 points)

### Q13 (2 pts)
What is "privatization" in the context of GPU histograms?

A) Making data private to each thread  
B) Each block has its own histogram in shared memory  
C) Encrypting the histogram data  
D) Using private GPU memory  

---

### Q14 (2 pts)
In a privatized histogram, when do we use global atomics?

A) For every element  
B) Never - we avoid global atomics entirely  
C) Only when merging block histograms to global  
D) Only for overflow bins  

---

### Q15 (2 pts)
For a histogram with 256 bins and 256 blocks, how many global atomic operations per bin?

A) 1  
B) 256 (one per block)  
C) 65,536 (256 × 256)  
D) Depends on data size  

---

## Bonus Question (2 pts)

### Q16 (2 pts)
You need to compute both sum and count (for mean calculation) in one pass. What's the most efficient approach?

A) Run sum reduction, then count reduction (two passes)  
B) Use two separate atomic counters  
C) Accumulate both sum and count in a single reduction kernel  
D) Count is just N, only compute sum  

---

## Answer Key

<details>
<summary>Click to reveal answers</summary>

| Q | Answer | Explanation |
|---|--------|-------------|
| 1 | B | Tree reduction has log₂(N) parallel steps |
| 2 | B | Ensure data consistency before next step |
| 3 | C | log₂(256) = 8 steps |
| 4 | C | Each block produces one partial result |
| 5 | B | Warp = 32 threads executing in lockstep |
| 6 | B | shfl_down reads from higher lane numbers |
| 7 | B | Threads in warp are implicitly synchronized |
| 8 | C | log₂(32) = 5 shuffle operations |
| 9 | B | Atomics prevent race conditions |
| 10 | B | Atomics return the old (pre-operation) value |
| 11 | B | All threads compete for same memory location |
| 12 | B | Shared memory atomics are ~10x faster |
| 13 | B | Each block has private histogram in shared mem |
| 14 | C | One global atomic per bin during merge phase |
| 15 | B | Each block contributes one atomic per bin |
| 16 | C | Fuse sum and count in single reduction |

**Scoring:**
- 30-32: Excellent! Ready for Week 5
- 24-29: Good understanding, review weaker areas
- 18-23: Review notebooks before proceeding
- <18: Revisit Week 4 materials

</details>

---

## Self-Reflection

After completing this quiz, consider:

1. **Reduction patterns:** Can you trace through a tree reduction step by step?

2. **Warp primitives:** Do you understand when to use shfl_down vs shfl_xor?

3. **Atomic trade-offs:** Can you identify when atomics are appropriate vs reduction?

4. **Histogram optimization:** Can you explain why privatization helps performance?

---

## Practical Skills Checklist

Before moving to Week 5, ensure you can:

- [ ] Implement sum, max, min reduction kernels
- [ ] Use warp shuffle for 32-element reduction
- [ ] Choose between atomics and reduction for a given problem
- [ ] Implement histogram with shared memory privatization
- [ ] Calculate reduction step count for given thread count

---

## Ready for Week 5?

If you scored 24+, you're ready for **Week 5: Matrix Operations**!

Preview:
- Matrix-vector multiplication
- Tiled matrix-matrix multiplication
- Memory coalescing in 2D
- Cache blocking strategies
