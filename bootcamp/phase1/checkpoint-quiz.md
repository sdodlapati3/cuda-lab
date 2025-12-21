# Phase 1 Checkpoint Quizzes

Quick self-assessment quizzes to verify understanding before moving to Phase 2.

---

## Week 5: Execution Model

### Quiz (10 questions)

1. **How many threads are in a warp?**
   - [ ] A) 16
   - [ ] B) 32
   - [ ] C) 64
   - [ ] D) 128

2. **What determines maximum threads per block?**
   - [ ] A) Array size
   - [ ] B) GPU compute capability
   - [ ] C) Compiler version
   - [ ] D) Host memory

3. **`threadIdx.x` gives:**
   - [ ] A) Global thread index
   - [ ] B) Thread index within block
   - [ ] C) Block index
   - [ ] D) Warp index

4. **Grid-stride loops are used when:**
   - [ ] A) Array size < threads launched
   - [ ] B) Array size > threads launched
   - [ ] C) Always
   - [ ] D) Never

5. **Warp divergence occurs when:**
   - [ ] A) All threads take same branch
   - [ ] B) Threads in a warp take different branches
   - [ ] C) Memory access is coalesced
   - [ ] D) Kernel launches fail

6. **SIMT stands for:**
   - [ ] A) Single Instruction Multiple Threads
   - [ ] B) Simple Instruction Machine Thread
   - [ ] C) Synchronized Instruction Memory Transfer
   - [ ] D) Serial Input Multiple Tasks

7. **Occupancy is:**
   - [ ] A) Memory usage
   - [ ] B) Active warps / Maximum warps per SM
   - [ ] C) Thread count
   - [ ] D) Bandwidth

8. **CUDA streams allow:**
   - [ ] A) File I/O
   - [ ] B) Concurrent operations (kernels, copies)
   - [ ] C) Debugging
   - [ ] D) Compilation

9. **CUDA events are used for:**
   - [ ] A) Memory allocation
   - [ ] B) Timing and synchronization
   - [ ] C) Thread indexing
   - [ ] D) Error handling

10. **The default stream (0) is:**
    - [ ] A) Asynchronous with all streams
    - [ ] B) Synchronous with all streams (legacy behavior)
    - [ ] C) Fastest stream
    - [ ] D) Only for memory copies

<details>
<summary>Answers</summary>

1. B - 32
2. B - GPU compute capability
3. B - Thread index within block
4. B - Array size > threads launched
5. B - Threads in a warp take different branches
6. A - Single Instruction Multiple Threads
7. B - Active warps / Maximum warps per SM
8. B - Concurrent operations (kernels, copies)
9. B - Timing and synchronization
10. B - Synchronous with all streams (legacy behavior)

</details>

---

## Week 6: Memory Hierarchy

### Quiz (10 questions)

1. **Which memory is fastest?**
   - [ ] A) Global memory
   - [ ] B) Shared memory
   - [ ] C) Registers
   - [ ] D) Constant memory

2. **Global memory access is coalesced when:**
   - [ ] A) Threads access random locations
   - [ ] B) Consecutive threads access consecutive addresses
   - [ ] C) All threads access same address
   - [ ] D) Memory is pinned

3. **Shared memory is visible to:**
   - [ ] A) All threads in grid
   - [ ] B) All threads in block
   - [ ] C) Single thread only
   - [ ] D) Host and device

4. **Bank conflicts occur in:**
   - [ ] A) Global memory
   - [ ] B) Shared memory
   - [ ] C) Registers
   - [ ] D) Constant memory

5. **How many shared memory banks (modern GPUs)?**
   - [ ] A) 16
   - [ ] B) 32
   - [ ] C) 64
   - [ ] D) 128

6. **Register spilling means:**
   - [ ] A) Registers are freed
   - [ ] B) Values overflow to slower local memory
   - [ ] C) Faster execution
   - [ ] D) Compilation error

7. **Constant memory is best for:**
   - [ ] A) Large arrays
   - [ ] B) Read-only data accessed uniformly
   - [ ] C) Per-thread data
   - [ ] D) Output arrays

8. **L1/L2 caches are:**
   - [ ] A) Programmer-managed
   - [ ] B) Hardware-managed (automatic)
   - [ ] C) Not present on GPUs
   - [ ] D) Only for texture memory

9. **To avoid bank conflicts, use:**
   - [ ] A) Sequential access
   - [ ] B) Padding to shift access patterns
   - [ ] C) More threads
   - [ ] D) Less shared memory

10. **AoS vs SoA: which is better for coalescing?**
    - [ ] A) Array of Structures (AoS)
    - [ ] B) Structure of Arrays (SoA)
    - [ ] C) Both equal
    - [ ] D) Depends on compiler

<details>
<summary>Answers</summary>

1. C - Registers
2. B - Consecutive threads access consecutive addresses
3. B - All threads in block
4. B - Shared memory
5. B - 32
6. B - Values overflow to slower local memory
7. B - Read-only data accessed uniformly
8. B - Hardware-managed (automatic)
9. B - Padding to shift access patterns
10. B - Structure of Arrays (SoA)

</details>

---

## Week 7: First Real Kernels

### Quiz (10 questions)

1. **Vector add is typically:**
   - [ ] A) Compute-bound
   - [ ] B) Memory-bound
   - [ ] C) Launch-bound
   - [ ] D) CPU-bound

2. **SAXPY stands for:**
   - [ ] A) Single Array X Plus Y
   - [ ] B) Scalar A times X Plus Y
   - [ ] C) Sequential Array X Product Y
   - [ ] D) Shared Array X Plus Y

3. **Parallel reduction requires:**
   - [ ] A) No synchronization
   - [ ] B) Synchronization between reduction steps
   - [ ] C) Only global memory
   - [ ] D) CPU involvement

4. **Warp shuffle reduction is faster because:**
   - [ ] A) Uses global memory
   - [ ] B) No shared memory needed within warp
   - [ ] C) More threads
   - [ ] D) Simpler code

5. **Inclusive scan: `[1,2,3,4]` → ?**
   - [ ] A) `[0,1,3,6]`
   - [ ] B) `[1,3,6,10]`
   - [ ] C) `[1,2,3,4]`
   - [ ] D) `[4,3,2,1]`

6. **Exclusive scan: `[1,2,3,4]` → ?**
   - [ ] A) `[0,1,3,6]`
   - [ ] B) `[1,3,6,10]`
   - [ ] C) `[1,2,3,4]`
   - [ ] D) `[0,0,0,0]`

7. **Histogram with global atomics is slow because:**
   - [ ] A) Too much shared memory
   - [ ] B) High contention on same bins
   - [ ] C) Not enough threads
   - [ ] D) Coalescing issues

8. **Matrix multiply is typically:**
   - [ ] A) Memory-bound (naive)
   - [ ] B) Compute-bound (optimized)
   - [ ] C) Always memory-bound
   - [ ] D) Never benefits from tiling

9. **Tiled matrix multiply uses shared memory to:**
   - [ ] A) Store final result
   - [ ] B) Cache input tiles for reuse
   - [ ] C) Communicate between blocks
   - [ ] D) Avoid registers

10. **Before writing a reduction kernel, first check:**
    - [ ] A) If CPU can do it
    - [ ] B) If CUB::DeviceReduce exists
    - [ ] C) If global memory is available
    - [ ] D) Compiler version

<details>
<summary>Answers</summary>

1. B - Memory-bound
2. B - Scalar A times X Plus Y
3. B - Synchronization between reduction steps
4. B - No shared memory needed within warp
5. B - `[1,3,6,10]`
6. A - `[0,1,3,6]`
7. B - High contention on same bins
8. B - Compute-bound (optimized)
9. B - Cache input tiles for reuse
10. B - If CUB::DeviceReduce exists (library-first!)

</details>

---

## Week 8: Synchronization & Atomics

### Quiz (10 questions)

1. **`__syncthreads()` synchronizes:**
   - [ ] A) All threads in grid
   - [ ] B) All threads in block
   - [ ] C) All threads in warp
   - [ ] D) Host and device

2. **Calling `__syncthreads()` in divergent code can cause:**
   - [ ] A) Faster execution
   - [ ] B) Deadlock
   - [ ] C) Memory leak
   - [ ] D) Compilation error

3. **`__shfl_down_sync` is used for:**
   - [ ] A) Global memory access
   - [ ] B) Warp-level data exchange
   - [ ] C) Block synchronization
   - [ ] D) Host-device copy

4. **`atomicAdd` guarantees:**
   - [ ] A) Ordering across all threads
   - [ ] B) Read-modify-write atomicity
   - [ ] C) No performance impact
   - [ ] D) Coalesced access

5. **`atomicCAS` stands for:**
   - [ ] A) Atomic Cache And Store
   - [ ] B) Atomic Compare And Swap
   - [ ] C) Atomic Count And Sum
   - [ ] D) Atomic Copy And Sync

6. **To reduce atomic contention:**
   - [ ] A) Use more atomics
   - [ ] B) Privatize (per-thread/warp accumulation first)
   - [ ] C) Use global memory
   - [ ] D) Remove synchronization

7. **`__threadfence()` ensures:**
   - [ ] A) All threads sync
   - [ ] B) Prior writes visible to other threads
   - [ ] C) Kernel completion
   - [ ] D) Error clearing

8. **Cooperative groups provide:**
   - [ ] A) Only block-level sync
   - [ ] B) Flexible sync at any granularity
   - [ ] C) CPU-GPU sync
   - [ ] D) File I/O

9. **Lock-free algorithms use:**
   - [ ] A) Spinlocks
   - [ ] B) CAS loops that always make progress
   - [ ] C) `__syncthreads()`
   - [ ] D) Host synchronization

10. **Spinlocks on GPU are dangerous because:**
    - [ ] A) Too fast
    - [ ] B) Warp lockstep can cause deadlock
    - [ ] C) Not supported
    - [ ] D) Use too much memory

<details>
<summary>Answers</summary>

1. B - All threads in block
2. B - Deadlock
3. B - Warp-level data exchange
4. B - Read-modify-write atomicity
5. B - Atomic Compare And Swap
6. B - Privatize (per-thread/warp accumulation first)
7. B - Prior writes visible to other threads
8. B - Flexible sync at any granularity
9. B - CAS loops that always make progress
10. B - Warp lockstep can cause deadlock

</details>

---

## Phase 1 Gate Checklist

Before proceeding to Phase 2, verify you can:

- [ ] Explain the thread hierarchy (thread → warp → block → grid)
- [ ] Calculate global thread index for 1D, 2D, 3D grids
- [ ] Identify coalesced vs non-coalesced memory access
- [ ] Know when to use shared memory vs registers
- [ ] Write a reduction achieving >70% memory bandwidth
- [ ] Explain *why* a kernel is slow (memory-bound vs compute-bound)
- [ ] Use CUB/cuBLAS when appropriate (library-first thinking)

---

## Scoring

- **9-10 per week:** Ready to proceed
- **7-8 per week:** Review weak areas, then proceed
- **<7 per week:** Re-study the material before Phase 2

**Total Phase 1: 40 questions**
- **36-40:** Excellent understanding
- **32-35:** Good, minor review needed
- **<32:** Spend more time on Phase 1 before continuing
