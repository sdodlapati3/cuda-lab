# Week 9 Checkpoint Quiz: CUDA Streams & Concurrency

## Conceptual Questions

### Q1: Stream Basics
What is the default stream and how does it differ from non-default streams?

**Your Answer:**

---

### Q2: Async Operations
Why is pinned (page-locked) memory required for cudaMemcpyAsync?

**Your Answer:**

---

### Q3: Overlap Opportunities
List the three types of operations that can potentially overlap on modern GPUs.

**Your Answer:**

---

### Q4: Stream Priorities
When would you use stream priorities, and what are the limitations?

**Your Answer:**

---

### Q5: Events
How do CUDA events enable inter-stream synchronization?

**Your Answer:**

---

## Code Analysis

### Q6: Find the Issue
```cpp
float *d_data;
cudaMalloc(&d_data, size);

cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
kernel<<<blocks, threads, 0, stream>>>(d_data, n);

// Use result immediately
printf("First element: %f\n", h_data[0]);  // Problem?
```

What's wrong with this code?

**Your Answer:**

---

### Q7: Overlap Pattern
```cpp
// Version A
for (int i = 0; i < N_CHUNKS; i++) {
    cudaMemcpyAsync(d_data, h_data + i*chunk, chunk_size, H2D, stream);
    kernel<<<grid, block, 0, stream>>>(d_data, chunk);
    cudaMemcpyAsync(h_out + i*chunk, d_out, chunk_size, D2H, stream);
}

// Version B
for (int i = 0; i < N_CHUNKS; i++) {
    cudaMemcpyAsync(d_data[i], h_data + i*chunk, chunk_size, H2D, streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(d_data[i], chunk);
    cudaMemcpyAsync(h_out + i*chunk, d_out[i], chunk_size, D2H, streams[i]);
}
```

Which version enables overlap? Why?

**Your Answer:**

---

## Practical Exercises

### Exercise 1: Stream Pipeline
Implement a 4-stage pipeline:
1. Copy chunk N from host to device
2. Process chunk N-1
3. Copy chunk N-2 from device to host

```cpp
// Your implementation:
```

### Exercise 2: Event Timing
Write code to accurately time a kernel using CUDA events:

```cpp
// Your implementation:
```

### Exercise 3: Multi-Stream Reduction
Implement a reduction where each stream handles a portion of the array, then combine results:

```cpp
// Your implementation:
```

---

## Profiling Challenge

Run `nsys profile` on your stream code and answer:
1. Are H2D and D2H transfers overlapping with compute?
2. How many streams are executing concurrently?
3. What is the GPU utilization percentage?

**Your Analysis:**

---

## Self-Assessment

Rate your understanding (1-5):
- [ ] Creating and managing streams: ___
- [ ] Pinned memory for async: ___
- [ ] Overlap patterns: ___
- [ ] Event-based synchronization: ___
- [ ] Profiling concurrency: ___

**Total Score: ___ / 25**

**Areas to Review:**
