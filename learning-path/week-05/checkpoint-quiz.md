# Week 5 Checkpoint Quiz: Prefix Sum (Scan)

Test your understanding of scan algorithms and applications.

---

## Part 1: Conceptual Questions (12 points)

### Q1. Scan Types (3 points)
Given input `[2, 4, 1, 3]`, what is:
a) The inclusive scan result?
b) The exclusive scan result?
c) How are they related?

### Q2. Algorithm Comparison (3 points)
Compare Hillis-Steele and Blelloch algorithms:
a) Which has more work complexity?
b) Which has fewer parallel steps?
c) Which is work-efficient?

### Q3. Bank Conflicts (3 points)
Why do naive scan implementations suffer from bank conflicts? How do we avoid them?

### Q4. Large Arrays (3 points)
When scanning arrays larger than a single block:
a) What is the 3-pass approach?
b) Why is the middle pass necessary?
c) What is the work complexity?

---

## Part 2: Code Analysis (10 points)

### Q5. Identify the Bug (5 points)
```cpp
__global__ void inclusive_scan(int* data, int n) {
    __shared__ int temp[256];
    int tid = threadIdx.x;
    
    temp[tid] = data[tid];
    __syncthreads();
    
    for (int stride = 1; stride < n; stride *= 2) {
        int val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;  // BUG HERE
        __syncthreads();
    }
    
    data[tid] = temp[tid];
}
```
What's wrong with this code? How would you fix it?

### Q6. Complete the Code (5 points)
Complete the downsweep phase of Blelloch scan:
```cpp
// After upsweep, temp[n-1] contains total sum
// Set last element to 0 for exclusive scan
temp[n - 1] = 0;

for (int stride = n / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
        int ai = /* YOUR CODE */;
        int bi = /* YOUR CODE */;
        
        int t = temp[ai];
        temp[ai] = /* YOUR CODE */;
        temp[bi] = /* YOUR CODE */;
    }
}
```

---

## Part 3: Application Problems (8 points)

### Q7. Stream Compaction (4 points)
Given array `[3, 0, 5, 0, 0, 2, 0, 1]`:
a) What is the predicate array (non-zero elements)?
b) What is the exclusive scan of predicates?
c) What is the final compacted array?
d) How many elements are in the result?

### Q8. Radix Sort Setup (4 points)
For radix sort of `[5, 3, 7, 2]` (3-bit numbers):
a) What are the predicates for bit 0 (LSB)?
b) How does scan help with the "0" bucket placement?
c) How does scan help with the "1" bucket placement?

---

## Answers

<details>
<summary>Click to reveal answers</summary>

### Q1 Answers
a) `[2, 6, 7, 10]` - running sum including current
b) `[0, 2, 6, 7]` - running sum excluding current
c) Exclusive scan = inclusive scan shifted right with 0 prepended

### Q2 Answers
a) Hillis-Steele: O(n log n) work vs Blelloch: O(n) work
b) Hillis-Steele: O(log n) vs Blelloch: O(2 log n)
c) Blelloch is work-efficient

### Q3 Answers
Bank conflicts occur when multiple threads access same bank. Sequential addressing causes conflicts. Solution: Add padding (e.g., `CONFLICT_FREE_OFFSET(i)`)

### Q4 Answers
a) Pass 1: Block-level scan. Pass 2: Scan of block sums. Pass 3: Add block sums to blocks
b) To propagate sums between blocks
c) Still O(n) work

### Q5 Answer
Bug: Reading and writing to same array (temp[tid]) in same iteration causes race condition. Fix: Use double buffering or ensure read happens before any write.

### Q6 Answer
```cpp
int ai = stride * (2 * tid + 1) - 1;
int bi = stride * (2 * tid + 2) - 1;

int t = temp[ai];
temp[ai] = temp[bi];
temp[bi] += t;
```

### Q7 Answers
a) Predicates: `[1, 0, 1, 0, 0, 1, 0, 1]`
b) Exclusive scan: `[0, 1, 1, 2, 2, 2, 3, 3]`
c) Compacted: `[3, 5, 2, 1]`
d) 4 elements

### Q8 Answers
a) Bit 0 predicates: 5=101→1, 3=011→1, 7=111→1, 2=010→0 → `[1, 1, 1, 0]`
b) Scan of "0" predicates gives destination indices for 0-bucket
c) Scan of "1" predicates + bucket_0_count gives destination indices for 1-bucket

</details>

---

## Scoring

| Part | Points | Your Score |
|------|--------|------------|
| Conceptual | 12 | |
| Code Analysis | 10 | |
| Applications | 8 | |
| **Total** | **30** | |

**Passing Score: 25/30**
