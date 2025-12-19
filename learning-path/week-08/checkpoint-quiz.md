# Week 8 Checkpoint Quiz: Profiling & Analysis

## Instructions
- Total points: 30
- Time: 30 minutes
- Passing score: 24/30 (80%)

---

## Part 1: Nsight Compute (10 points)

### Question 1 (3 points)
What is the difference between Nsight Compute and Nsight Systems? When would you use each?

**Your answer:**

---

### Question 2 (3 points)
Explain what each of these Nsight Compute metrics indicates:
- SM Throughput (%)
- Memory Throughput (%)
- Achieved Occupancy

**Your answer:**

---

### Question 3 (4 points)
A kernel shows these metrics:
- SM Throughput: 85%
- Memory Throughput: 20%
- Achieved Occupancy: 75%

What type of bottleneck does this indicate? What optimizations would you try?

**Your answer:**

---

## Part 2: Roofline Analysis (10 points)

### Question 4 (3 points)
Define arithmetic intensity. What units is it measured in?

**Your answer:**

---

### Question 5 (4 points)
For a GEMM (matrix multiply) of two 1024×1024 matrices:
- Calculate the total FLOPs
- Calculate the memory traffic (minimum)
- Calculate the arithmetic intensity

**Your answer:**

---

### Question 6 (3 points)
On a GPU with:
- Peak compute: 10 TFLOPS
- Peak memory bandwidth: 500 GB/s

What is the arithmetic intensity threshold between compute-bound and memory-bound?

**Your answer:**

---

## Part 3: Bottleneck Analysis (10 points)

### Question 7 (4 points)
Match each symptom to its likely bottleneck:

Symptoms:
A. High SM throughput, low memory throughput
B. Low SM throughput, high memory throughput
C. Low occupancy, many spill loads
D. Large gaps in GPU timeline

Bottlenecks:
1. Register pressure
2. Compute-bound
3. Memory-bound
4. CPU bottleneck or synchronization

**Your answer:**

---

### Question 8 (3 points)
What causes "warp stall" and how do you identify it in the profiler?

**Your answer:**

---

### Question 9 (3 points)
A profiler shows your kernel has 95% L1 cache hit rate but only achieves 30% of peak bandwidth. What might be wrong?

**Your answer:**

---

## Bonus Question (2 points)
Explain what "latency hiding" means and how occupancy helps with it.

**Your answer:**

---

## Answer Key

### Q1: Nsight Compute vs Systems
- **Nsight Compute**: Kernel-level profiler, detailed metrics per kernel
  - Use for: optimizing individual kernels
- **Nsight Systems**: Application-level profiler, timeline view
  - Use for: understanding CPU-GPU interaction, finding synchronization issues

### Q2: Metrics Explanation
- **SM Throughput**: Percentage of SM compute cycles used (higher = more compute utilization)
- **Memory Throughput**: Percentage of peak memory bandwidth used
- **Achieved Occupancy**: Actual warps executing / max warps per SM

### Q3: Bottleneck Analysis
- High SM (85%), low memory (20%) = **Compute-bound**
- Optimizations:
  1. Increase parallelism (more threads)
  2. Reduce instruction latency (less divergence)
  3. Use faster instructions (FMA, intrinsics)
  4. Unroll loops for ILP

### Q4: Arithmetic Intensity
- Definition: FLOPs / Bytes transferred
- Units: FLOP/byte

### Q5: GEMM Calculations
- FLOPs: 2 × 1024³ = 2,147,483,648 ≈ 2.1 GFLOP
- Memory: 3 × 1024² × 4 bytes = 12.6 MB (two inputs + one output)
- AI = 2.1 × 10⁹ / (12.6 × 10⁶) ≈ 170 FLOP/byte

### Q6: Roofline Threshold
- Ridge point = Peak Compute / Peak Bandwidth
- = 10 TFLOPS / 500 GB/s = 20 FLOP/byte
- Below 20: memory-bound
- Above 20: compute-bound

### Q7: Matching
- A → 2 (Compute-bound)
- B → 3 (Memory-bound)
- C → 1 (Register pressure)
- D → 4 (CPU bottleneck)

### Q8: Warp Stalls
- Cause: Warps waiting for resources (memory, execution units)
- Identification: Warp Stall Reasons in Nsight Compute
  - stall_memory: waiting for memory
  - stall_math: waiting for compute
  - stall_sync: waiting for barrier

### Q9: Cache Hit but Low Bandwidth
- Possible causes:
  1. Low occupancy (not enough requests)
  2. Small working set (not stressing bandwidth)
  3. Compute-bound (not memory-limited)
  4. Bank conflicts in shared memory

### Bonus: Latency Hiding
- Memory has high latency (~400 cycles)
- While one warp waits for memory, GPU switches to other warps
- More warps (higher occupancy) = better latency hiding
- Like having multiple tasks - switch when one blocks
