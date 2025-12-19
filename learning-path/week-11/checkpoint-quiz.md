# Week 11 Checkpoint Quiz: Cooperative Groups & Dynamic Parallelism

## Instructions
Complete this quiz after finishing Days 1-4. Aim for 80% or higher.

---

## Section A: Cooperative Groups (10 questions)

### Question 1
What header is required for cooperative groups?
- A) `<cuda_runtime.h>`
- B) `<cooperative_groups.h>`
- C) `<cg_groups.h>`
- D) `<thread_groups.h>`

### Question 2
What does `cg::this_thread_block()` return?
- A) A grid group
- B) A tile group
- C) A thread block group
- D) A coalesced group

### Question 3
What size tile can you create with `cg::tiled_partition<T>(tb)`?
- A) Any size
- B) Powers of 2 up to 32
- C) Only 32
- D) Powers of 2 up to warp size

### Question 4
What is the purpose of a coalesced group?
- A) Memory coalescing
- B) Grouping threads that are active in divergent code
- C) Grouping tiles together
- D) Grid-level synchronization

### Question 5
How do you synchronize within a tile group?
- A) `__syncthreads()`
- B) `tile.sync()`
- C) `cudaDeviceSynchronize()`
- D) `tile.wait()`

### Question 6
What is `cg::grid_group` used for?
- A) Grouping multiple thread blocks
- B) Creating 3D grids
- C) Full grid synchronization
- D) Dynamic grid sizing

### Question 7
What special launch is required for grid-wide synchronization?
- A) `<<<blocks, threads>>>`
- B) `cudaLaunchCooperativeKernel()`
- C) `cudaLaunchGrid()`
- D) `cudaLaunchSynchronized()`

### Question 8
What limits grid size for cooperative launches?
- A) No limits
- B) Maximum active blocks per SM
- C) 1024 blocks total
- D) Shared memory only

### Question 9
What does `tile.shfl()` do?
- A) Shuffles data within the tile
- B) Shuffles tiles
- C) Shuffles warps
- D) Shuffles blocks

### Question 10
Can coalesced groups span multiple warps?
- A) Yes, always
- B) No, never
- C) Only with special flags
- D) Only in CUDA 12+

---

## Section B: Dynamic Parallelism (10 questions)

### Question 11
What is dynamic parallelism?
- A) Dynamically changing grid size
- B) Launching kernels from within kernels
- C) Dynamic memory allocation
- D) Parallel file I/O

### Question 12
What compute capability is required for dynamic parallelism?
- A) 2.0+
- B) 3.0+
- C) 3.5+
- D) 5.0+

### Question 13
How do you synchronize child kernels in dynamic parallelism?
- A) `__syncthreads()`
- B) `cudaDeviceSynchronize()` in kernel
- C) `cudaStreamSynchronize()`
- D) Automatic synchronization

### Question 14
What is the default stream behavior for child kernels?
- A) All child kernels use stream 0
- B) Each child gets a unique stream
- C) Children share parent's stream
- D) No streams on device

### Question 15
What linker flag is needed for dynamic parallelism?
- A) `-dynamic`
- B) `-rdc=true`
- C) `-dp=true`
- D) `-parallel`

### Question 16
What happens when parent kernel returns before child completes?
- A) Child is terminated
- B) Undefined behavior
- C) Child continues but results may not be visible
- D) Error is raised

### Question 17
What is the nesting depth limit for dynamic parallelism?
- A) 2 levels
- B) 24 levels
- C) Unlimited
- D) 8 levels

### Question 18
Which is NOT a valid use case for dynamic parallelism?
- A) Quicksort recursion
- B) Adaptive mesh refinement
- C) Simple vector addition
- D) Tree traversal

### Question 19
How do child kernels access parent's local memory?
- A) Directly through pointers
- B) Through global memory only
- C) Through shared memory
- D) Through registers

### Question 20
What is the overhead of dynamic parallelism?
- A) Zero overhead
- B) Higher than host launches
- C) Same as host launches
- D) Lower than host launches

---

## Answer Key

### Section A: Cooperative Groups
1. B - `<cooperative_groups.h>`
2. C - A thread block group
3. D - Powers of 2 up to warp size
4. B - Grouping threads that are active in divergent code
5. B - `tile.sync()`
6. C - Full grid synchronization
7. B - `cudaLaunchCooperativeKernel()`
8. B - Maximum active blocks per SM
9. A - Shuffles data within the tile
10. B - No, never (coalesced groups are subsets of warps)

### Section B: Dynamic Parallelism
11. B - Launching kernels from within kernels
12. C - 3.5+
13. B - `cudaDeviceSynchronize()` in kernel
14. A - All child kernels use stream 0 (implicit NULL stream)
15. B - `-rdc=true` (relocatable device code)
16. C - Child continues but results may not be visible
17. B - 24 levels
18. C - Simple vector addition (no need for DP)
19. B - Through global memory only (local/shared not accessible)
20. B - Higher than host launches (device runtime overhead)

---

## Scoring
- 18-20 correct: Excellent! Ready for Week 12
- 14-17 correct: Good understanding, review missed concepts
- 10-13 correct: Review Days 1-4 materials
- Below 10: Revisit the week's content before proceeding
