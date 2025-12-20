# Week 18 Checkpoint Quiz: MIG & Production CUDA

## Section A: Multi-Instance GPU (5 questions)

### Q1: MIG Capability
Which NVIDIA GPUs support MIG?
- A) All GeForce RTX cards
- B) Only A100 and H100 datacenter GPUs
- C) Any GPU with compute capability 8.0+
- D) Only Quadro professional cards

<details>
<summary>Answer</summary>
B) Only A100 and H100 (and newer datacenter GPUs like H200, B100, B200). MIG is a datacenter feature, not available on consumer GPUs.
</details>

---

### Q2: MIG Architecture
What does MIG partition?
- A) Only GPU memory
- B) Only streaming multiprocessors (SMs)
- C) Memory, SMs, and L2 cache in hardware
- D) Time-slicing of the GPU

<details>
<summary>Answer</summary>
C) Memory, SMs, and L2 cache in hardware. MIG provides true hardware partitioning, not time-slicing.
</details>

---

### Q3: GPU Instance vs Compute Instance
What is the relationship between GPU Instance and Compute Instance?
- A) They are the same thing
- B) GPU Instance contains one or more Compute Instances
- C) Compute Instance contains one or more GPU Instances
- D) They are independent resources

<details>
<summary>Answer</summary>
B) GPU Instance contains one or more Compute Instances. A GPU Instance defines memory partition, then can be subdivided into Compute Instances for SM allocation.
</details>

---

### Q4: Targeting MIG Instances
How do you run code on a specific MIG instance?
- A) Use cudaSetDevice() with instance number
- B) Set CUDA_VISIBLE_DEVICES to the MIG UUID
- C) Call cudaMigSetInstance()
- D) MIG instances are automatically assigned

<details>
<summary>Answer</summary>
B) Set CUDA_VISIBLE_DEVICES to the MIG UUID (e.g., `MIG-12345678-...`). The application sees it as device 0.
</details>

---

### Q5: MIG Configuration
Who can typically configure MIG (create/destroy instances)?
- A) Any CUDA application
- B) Only root/administrator
- C) Any user with GPU access
- D) Only NVIDIA drivers

<details>
<summary>Answer</summary>
B) Only root/administrator. MIG configuration requires elevated privileges; users can only query and use existing instances.
</details>

---

## Section B: Error Management (5 questions)

### Q6: Sticky vs Non-Sticky Errors
Which error type requires cudaDeviceReset() to recover?
- A) cudaErrorMemoryAllocation
- B) cudaErrorInvalidValue
- C) cudaErrorIllegalAddress
- D) cudaErrorInvalidConfiguration

<details>
<summary>Answer</summary>
C) cudaErrorIllegalAddress. Illegal memory access is a sticky error that corrupts device state and requires reset.
</details>

---

### Q7: Error Checking
What is the difference between `cudaGetLastError()` and `cudaPeekAtLastError()`?
- A) No difference
- B) GetLastError clears the error, Peek does not
- C) Peek is faster
- D) GetLastError is deprecated

<details>
<summary>Answer</summary>
B) cudaGetLastError() clears the error after returning it; cudaPeekAtLastError() returns the error without clearing.
</details>

---

### Q8: Async Error Detection
When checking for kernel execution errors, why do we need cudaDeviceSynchronize()?
- A) To improve performance
- B) Kernel errors are asynchronous and not visible until sync
- C) Required by the CUDA API
- D) To clear the error state

<details>
<summary>Answer</summary>
B) Kernel execution is asynchronous. Errors during execution aren't reported until synchronization occurs.
</details>

---

### Q9: Stream Errors
If one CUDA stream encounters an error:
- A) All streams are affected
- B) The device becomes unusable
- C) Only that stream's operations fail
- D) The application crashes immediately

<details>
<summary>Answer</summary>
C) Only that stream's operations fail (for non-sticky errors). Stream errors are isolated to the stream.
</details>

---

### Q10: Debug Environment
What does CUDA_LAUNCH_BLOCKING=1 do?
- A) Blocks new kernel launches
- B) Makes all launches synchronous for debugging
- C) Enables MIG mode
- D) Increases performance

<details>
<summary>Answer</summary>
B) Makes all kernel launches synchronous. This helps locate which kernel caused an error, at the cost of performance.
</details>

---

## Section C: Production Patterns (5 questions)

### Q11: GPU Health Monitoring
What metrics should a production GPU health check include?
- A) Only temperature
- B) Only memory usage
- C) Temperature, memory, ECC errors, compute test
- D) Only power consumption

<details>
<summary>Answer</summary>
C) Comprehensive health checks should include temperature, memory availability, ECC errors, and a compute test to verify functionality.
</details>

---

### Q12: RAII Wrappers
Why use RAII wrappers for CUDA resources?
- A) Better performance
- B) Automatic cleanup prevents leaks on exceptions
- C) Required by CUDA 12+
- D) Smaller binary size

<details>
<summary>Answer</summary>
B) RAII ensures resources are freed even when exceptions occur, preventing memory leaks and resource exhaustion.
</details>

---

### Q13: GPU Failover
In a multi-GPU system, what should happen when a GPU fails?
- A) Application should crash
- B) Wait for GPU to recover
- C) Mark GPU unavailable, failover to another GPU
- D) Retry on same GPU indefinitely

<details>
<summary>Answer</summary>
C) Production systems should detect failure, mark the GPU as unavailable, and failover to a healthy GPU if possible.
</details>

---

### Q14: ECC Errors
Uncorrectable ECC errors indicate:
- A) Normal operation
- B) Memory corruption that may affect results
- C) Need for driver update
- D) Temperature issues

<details>
<summary>Answer</summary>
B) Uncorrectable ECC errors mean data in GPU memory may be corrupted. Results should not be trusted.
</details>

---

### Q15: Deployment Best Practice
Before deploying a CUDA application to production, you should:
- A) Only test on development GPU
- B) Run memory sanitizers and stress tests
- C) Skip error checking for performance
- D) Use global memory exclusively

<details>
<summary>Answer</summary>
B) Run compute-sanitizer (memcheck, racecheck), stress tests, and verify all error handling paths work correctly.
</details>

---

## Scoring

| Score | Level |
|-------|-------|
| 13-15 | Expert - Production-ready skills |
| 10-12 | Proficient - Ready for deployment |
| 7-9 | Developing - Review production patterns |
| < 7 | Beginner - Study materials again |

## Key Takeaways

1. **MIG is datacenter-only** - A100/H100/B200, not consumer GPUs
2. **Sticky errors require reset** - Illegal access, asserts, ECC
3. **RAII for resource management** - Prevents leaks on exceptions
4. **Monitor GPU health** - Temperature, memory, ECC, compute
5. **Plan for failures** - Implement failover and recovery
