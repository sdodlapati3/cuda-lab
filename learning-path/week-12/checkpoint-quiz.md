# Week 12 Checkpoint Quiz: Multi-GPU & Advanced Topics

## Instructions
Complete this quiz after finishing Days 1-4. This is the final assessment of the curriculum.

---

## Section A: Multi-GPU Programming (10 questions)

### Question 1
How do you select which GPU to use?
- A) `cudaSelectDevice(id)`
- B) `cudaSetDevice(id)`
- C) `cudaUseDevice(id)`
- D) `cudaDeviceSet(id)`

### Question 2
What does `cudaGetDeviceCount` return?
- A) Number of CUDA cores
- B) Number of available GPUs
- C) Device compute capability
- D) Available memory

### Question 3
What is required for peer-to-peer (P2P) memory access?
- A) Same GPU model
- B) Unified Memory
- C) PCIe or NVLink connection
- D) CUDA 12+

### Question 4
What function enables direct GPU-to-GPU memory access?
- A) `cudaEnablePeer()`
- B) `cudaDeviceEnablePeerAccess()`
- C) `cudaP2PEnable()`
- D) `cudaSetPeerAccess()`

### Question 5
How do you copy memory directly between two GPUs?
- A) `cudaMemcpy()` with special flag
- B) `cudaMemcpyPeer()`
- C) `cudaP2PCopy()`
- D) Copy through host only

### Question 6
What happens if you access device memory from wrong GPU without P2P?
- A) Automatic transfer
- B) Error or undefined behavior
- C) Silent performance degradation
- D) Works normally

### Question 7
What is NVLink?
- A) A programming API
- B) High-speed GPU interconnect
- C) Memory type
- D) Compiler flag

### Question 8
What is a common multi-GPU pattern for domain decomposition?
- A) Each GPU handles random elements
- B) Each GPU handles contiguous chunk
- C) All GPUs duplicate all data
- D) Only one GPU does computation

### Question 9
What is "halo exchange" in multi-GPU programming?
- A) GPU-to-CPU communication
- B) Sharing boundary data between GPUs
- C) Memory allocation pattern
- D) Error handling

### Question 10
With Unified Memory on multi-GPU, what does `cudaMemAdvise` do?
- A) Allocates memory
- B) Provides hints about data access patterns
- C) Frees memory
- D) Synchronizes devices

---

## Section B: Advanced Optimization Review (10 questions)

### Question 11
What is the most important optimization in CUDA?
- A) Reducing register usage
- B) Memory coalescing
- C) Using shared memory
- D) All of the above, depends on bottleneck

### Question 12
What tool identifies whether code is compute or memory bound?
- A) cuda-memcheck
- B) Roofline analysis / Nsight Compute
- C) nvcc
- D) cuBLAS

### Question 13
What is occupancy?
- A) Ratio of active warps to maximum warps per SM
- B) Memory bandwidth usage
- C) GPU temperature
- D) Power consumption

### Question 14
When is low occupancy acceptable?
- A) Never
- B) When register pressure limits parallelism but improves ILP
- C) Only on old GPUs
- D) When using shared memory

### Question 15
What's the benefit of loop unrolling?
- A) Reduces code size
- B) Increases memory usage
- C) Reduces loop overhead, enables ILP
- D) Improves debugging

### Question 16
What is instruction-level parallelism (ILP)?
- A) Running multiple kernels
- B) Multiple independent operations per thread
- C) Using multiple GPUs
- D) Thread cooperation

### Question 17
What should you check first when optimizing CUDA code?
- A) Increase block size
- B) Profile to identify bottleneck
- C) Add more shared memory
- D) Use more registers

### Question 18
What is the L2 cache's role in CUDA?
- A) Stores kernel code only
- B) Caches global memory accesses for all SMs
- C) Stores shared memory
- D) Manages register allocation

### Question 19
When should you use constant memory?
- A) For large arrays
- B) For read-only data accessed by all threads
- C) For thread-local data
- D) For atomic operations

### Question 20
What is kernel fusion?
- A) Splitting one kernel into multiple
- B) Combining multiple kernels into one
- C) Running kernels in parallel
- D) Compiling kernels together

---

## Section C: Comprehensive Review (10 questions)

### Question 21
What's the execution unit for SIMT?
- A) Block
- B) Warp (32 threads)
- C) Grid
- D) Thread

### Question 22
What happens during warp divergence?
- A) Threads execute different branches serially
- B) Program crashes
- C) Automatic optimization
- D) Memory corruption

### Question 23
Name three memory types in order of speed (fastest first):
- A) Global, Shared, Registers
- B) Registers, Shared, Global
- C) Shared, Registers, Global
- D) Global, Registers, Shared

### Question 24
What is the purpose of `__syncthreads()`?
- A) Sync all threads in grid
- B) Sync all threads in block
- C) Sync CPU and GPU
- D) Sync memory

### Question 25
What's the main advantage of CUDA Graphs?
- A) Easier programming
- B) Reduced launch overhead
- C) More memory
- D) Better debugging

### Question 26
When would you use streams?
- A) To run kernels sequentially
- B) To overlap compute and memory transfers
- C) To reduce memory usage
- D) For debugging

### Question 27
What is bank conflict in shared memory?
- A) Memory corruption
- B) Multiple threads accessing same bank
- C) Out of memory error
- D) Compilation error

### Question 28
What does atomicAdd do?
- A) Regular addition
- B) Thread-safe read-modify-write
- C) Fast memory copy
- D) Sync threads

### Question 29
What is a reduction operation?
- A) Decreasing array size
- B) Combining array elements to single value
- C) Memory deallocation
- D) Thread termination

### Question 30
What compute capability is needed for Tensor Cores?
- A) 3.5+
- B) 5.0+
- C) 7.0+ (Volta)
- D) Any CUDA GPU

---

## Answer Key

### Section A: Multi-GPU Programming
1. B - `cudaSetDevice(id)`
2. B - Number of available GPUs
3. C - PCIe or NVLink connection
4. B - `cudaDeviceEnablePeerAccess()`
5. B - `cudaMemcpyPeer()`
6. B - Error or undefined behavior
7. B - High-speed GPU interconnect
8. B - Each GPU handles contiguous chunk
9. B - Sharing boundary data between GPUs
10. B - Provides hints about data access patterns

### Section B: Advanced Optimization Review
11. D - All of the above, depends on bottleneck
12. B - Roofline analysis / Nsight Compute
13. A - Ratio of active warps to maximum warps per SM
14. B - When register pressure limits parallelism but improves ILP
15. C - Reduces loop overhead, enables ILP
16. B - Multiple independent operations per thread
17. B - Profile to identify bottleneck
18. B - Caches global memory accesses for all SMs
19. B - For read-only data accessed by all threads
20. B - Combining multiple kernels into one

### Section C: Comprehensive Review
21. B - Warp (32 threads)
22. A - Threads execute different branches serially
23. B - Registers, Shared, Global
24. B - Sync all threads in block
25. B - Reduced launch overhead
26. B - To overlap compute and memory transfers
27. B - Multiple threads accessing same bank
28. B - Thread-safe read-modify-write
29. B - Combining array elements to single value
30. C - 7.0+ (Volta)

---

## Scoring
- 27-30 correct: Excellent! You've mastered CUDA programming
- 22-26 correct: Strong understanding, minor gaps
- 17-21 correct: Good foundation, review weak areas
- Below 17: Review weeks 1-12 before advanced work

## Congratulations on Completing the 12-Week CUDA Curriculum! 🎉
