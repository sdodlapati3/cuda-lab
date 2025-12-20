# Week 14 Checkpoint Quiz: Real-World CUDA Applications

**Total Points: 30** | **Passing Score: 24 (80%)**

---

## Part 1: Kernel Fusion (10 points)

### Question 1 (2 points)
What is the primary benefit of kernel fusion?

- A) Reduced code complexity
- B) Reduced memory bandwidth usage
- C) Faster compilation
- D) Better error handling

### Question 2 (2 points)
How many global memory accesses does a fused kernel performing Add → Multiply → ReLU require?

- A) 6 (2 per operation)
- B) 4 (load inputs, store output)
- C) 3 (2 loads, 1 store)
- D) 2 (1 load, 1 store)

### Question 3 (3 points)
Which operations are good candidates for fusion?

- A) Large matrix multiplications
- B) Element-wise operations chains
- C) Convolutions with large kernels
- D) Scatter/gather operations

### Question 4 (3 points)
In a fused softmax kernel, what technique enables single-pass computation?

- A) Shared memory caching
- B) Online algorithm (running max/sum)
- C) Warp shuffle reduction
- D) Atomic operations

---

## Part 2: Attention Mechanisms (10 points)

### Question 5 (2 points)
What is the memory complexity of naive attention for sequence length N?

- A) O(N)
- B) O(N log N)
- C) O(N²)
- D) O(N³)

### Question 6 (3 points)
What is the key innovation of Flash Attention?

- A) Using Tensor Cores for attention
- B) Never materializing the full attention matrix
- C) Parallelizing across heads
- D) Using FP16 instead of FP32

### Question 7 (2 points)
Why is online softmax important for memory-efficient attention?

- A) It's faster to compute
- B) It enables streaming without full matrix
- C) It uses less registers
- D) It's more numerically stable

### Question 8 (3 points)
For causal (autoregressive) attention, masked positions should be set to:

- A) 0
- B) 1
- C) -infinity (before softmax)
- D) Very small positive number

---

## Part 3: PyTorch Extensions & Benchmarking (10 points)

### Question 9 (2 points)
Which header is required for PyTorch CUDA extensions?

- A) `<pytorch.h>`
- B) `<torch/extension.h>`
- C) `<torch/cuda.h>`
- D) `<pybind11/pybind.h>`

### Question 10 (2 points)
In a custom autograd.Function, which method must save tensors for backward?

- A) `forward()` using `ctx.save_for_backward()`
- B) `backward()` using `ctx.load_tensors()`
- C) `__init__()` using `self.saved_tensors`
- D) `apply()` using `save_tensors()`

### Question 11 (3 points)
What is wrong with this benchmarking code?

```cpp
auto t1 = std::chrono::now();
kernel<<<blocks, threads>>>(d_data, n);
auto t2 = std::chrono::now();
```

- A) Nothing, it's correct
- B) Missing cudaDeviceSynchronize() before t2
- C) Should use cudaMalloc instead
- D) Wrong timer resolution

### Question 12 (3 points)
A kernel achieves 400 GB/s on a GPU with 900 GB/s peak bandwidth. What is its efficiency?

- A) 25%
- B) 44%
- C) 55%
- D) 225%

---

## Answer Key

| Question | Answer | Explanation |
|----------|--------|-------------|
| 1 | B | Fusion eliminates intermediate memory writes/reads |
| 2 | C | Read A, read B, write C (intermediates stay in registers) |
| 3 | B | Element-wise chains can be fused trivially; GEMM is already compute-bound |
| 4 | B | Online softmax tracks running max and sum, enabling single-pass |
| 5 | C | Attention matrix is N×N, requiring O(N²) memory |
| 6 | B | Flash Attention tiles computation to avoid O(N²) memory |
| 7 | B | Can process Q, K, V in blocks without storing full attention matrix |
| 8 | C | -infinity ensures exp(-inf) = 0 after softmax |
| 9 | B | `<torch/extension.h>` provides all necessary PyTorch C++ APIs |
| 10 | A | `ctx.save_for_backward()` in forward, accessed via `ctx.saved_tensors` |
| 11 | B | Kernels are async; need sync before stopping timer |
| 12 | B | 400/900 ≈ 44% efficiency |

---

## Scoring Guide

- **30 points**: Excellent! Ready for production CUDA development
- **24-29 points**: Great understanding, review benchmarking section
- **18-23 points**: Good foundation, revisit attention optimization
- **Below 18 points**: Re-study the week's material before proceeding

---

## Congratulations! 🎉

You've completed the 14-week CUDA programming curriculum!

### What's Next?

1. **Apply your knowledge** - Build a real project
2. **Profile and optimize** - Use ncu and nsys on your code
3. **Explore advanced topics** - CUTLASS, cuDNN internals
4. **Contribute** - Share your learnings, write blog posts

### Recommended Projects

- Implement Flash Attention from scratch
- Build a custom PyTorch operator for your research
- Optimize an existing ML inference pipeline
- Create a CUDA-accelerated scientific simulation
