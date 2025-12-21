# Phase 0 Checkpoint Quizzes

Quick self-assessment quizzes to verify understanding before moving to Phase 1.

---

## Week 1: Build System Mastery

### Quiz (10 questions)

1. **What does `-arch=sm_80` specify?**
   - [ ] A) Optimization level
   - [ ] B) Target GPU compute capability
   - [ ] C) Number of threads per block
   - [ ] D) Memory allocation size

2. **What is PTX?**
   - [ ] A) A GPU hardware instruction set
   - [ ] B) An intermediate representation compiled to SASS
   - [ ] C) A debugging tool
   - [ ] D) A memory allocation API

3. **Why use `-lineinfo` flag?**
   - [ ] A) Faster execution
   - [ ] B) Smaller binary size
   - [ ] C) Source line mapping in profilers
   - [ ] D) Enable texture memory

4. **What build system is recommended for CUDA projects?**
   - [ ] A) Make only
   - [ ] B) CMake with Ninja
   - [ ] C) Autotools
   - [ ] D) Bazel only

5. **What does `CMAKE_CUDA_ARCHITECTURES` control?**
   - [ ] A) CPU architecture
   - [ ] B) GPU architectures to compile for
   - [ ] C) Memory alignment
   - [ ] D) Thread block size

6. **SASS is:**
   - [ ] A) Source code
   - [ ] B) Intermediate representation
   - [ ] C) GPU-specific machine code
   - [ ] D) A debugging format

7. **What flag enables maximum optimization?**
   - [ ] A) `-g`
   - [ ] B) `-O0`
   - [ ] C) `-O3`
   - [ ] D) `-debug`

8. **Why compile for multiple architectures?**
   - [ ] A) Faster compilation
   - [ ] B) Binary works on different GPU generations
   - [ ] C) Reduces binary size
   - [ ] D) Required for debugging

9. **What tool inspects PTX/SASS from a binary?**
   - [ ] A) nvprof
   - [ ] B) cuobjdump
   - [ ] C) cuda-gdb
   - [ ] D) nsys

10. **A reproducible benchmark harness should:**
    - [ ] A) Run once and report
    - [ ] B) Warm up GPU, run multiple trials, report statistics
    - [ ] C) Ignore cold start effects
    - [ ] D) Only measure kernel time

<details>
<summary>Answers</summary>

1. B - Target GPU compute capability (sm_80 = A100)
2. B - Intermediate representation compiled to SASS
3. C - Source line mapping in profilers
4. B - CMake with Ninja
5. B - GPU architectures to compile for
6. C - GPU-specific machine code
7. C - `-O3`
8. B - Binary works on different GPU generations
9. B - cuobjdump
10. B - Warm up GPU, run multiple trials, report statistics

</details>

---

## Week 2: Debugging Foundations

### Quiz (10 questions)

1. **What does compute-sanitizer detect?**
   - [ ] A) Performance issues
   - [ ] B) Race conditions and memory errors
   - [ ] C) Compilation warnings
   - [ ] D) Network issues

2. **How do you check for memory races?**
   - [ ] A) `compute-sanitizer --tool memcheck`
   - [ ] B) `compute-sanitizer --tool racecheck`
   - [ ] C) `cuda-gdb --race`
   - [ ] D) `nvprof --race`

3. **What is an "uninitialized read" error?**
   - [ ] A) Reading freed memory
   - [ ] B) Reading memory that was never written
   - [ ] C) Reading out of bounds
   - [ ] D) Reading from host in device code

4. **cuda-gdb is used for:**
   - [ ] A) Performance profiling
   - [ ] B) Interactive debugging with breakpoints
   - [ ] C) Memory allocation
   - [ ] D) Compilation

5. **Why is async error handling important?**
   - [ ] A) CUDA calls return immediately; errors appear later
   - [ ] B) It's faster
   - [ ] C) Required for multi-GPU
   - [ ] D) Debugging only

6. **`cudaGetLastError()` does what?**
   - [ ] A) Returns and clears the last error
   - [ ] B) Returns error without clearing
   - [ ] C) Throws an exception
   - [ ] D) Logs to file

7. **Nsight Systems is primarily for:**
   - [ ] A) Kernel-level metrics
   - [ ] B) System-wide timeline analysis
   - [ ] C) Memory debugging
   - [ ] D) Compilation

8. **What does a timeline show in Nsight Systems?**
   - [ ] A) Source code
   - [ ] B) GPU/CPU activity over time
   - [ ] C) Memory layout
   - [ ] D) PTX instructions

9. **What error indicates writing beyond array bounds?**
   - [ ] A) Race condition
   - [ ] B) Out-of-bounds access
   - [ ] C) Deadlock
   - [ ] D) Underflow

10. **Best practice after every kernel launch:**
    - [ ] A) Check error immediately
    - [ ] B) Never check errors (too slow)
    - [ ] C) Check errors in debug builds only
    - [ ] D) Wait 1 second

<details>
<summary>Answers</summary>

1. B - Race conditions and memory errors
2. B - `compute-sanitizer --tool racecheck`
3. B - Reading memory that was never written
4. B - Interactive debugging with breakpoints
5. A - CUDA calls return immediately; errors appear later
6. A - Returns and clears the last error
7. B - System-wide timeline analysis
8. B - GPU/CPU activity over time
9. B - Out-of-bounds access
10. A - Check error immediately (at least in debug)

</details>

---

## Week 3: Performance Analysis

### Quiz (10 questions)

1. **Nsight Compute (ncu) profiles:**
   - [ ] A) CPU code
   - [ ] B) Individual GPU kernels
   - [ ] C) Network traffic
   - [ ] D) Disk I/O

2. **A memory-bound kernel is limited by:**
   - [ ] A) Compute throughput
   - [ ] B) Memory bandwidth
   - [ ] C) Launch overhead
   - [ ] D) CPU speed

3. **Arithmetic intensity is:**
   - [ ] A) Number of threads
   - [ ] B) FLOPS / Bytes transferred
   - [ ] C) Memory bandwidth
   - [ ] D) Occupancy

4. **On a roofline chart, where do memory-bound kernels appear?**
   - [ ] A) On the flat (compute) ceiling
   - [ ] B) On the sloped (memory) line
   - [ ] C) Below the chart
   - [ ] D) Above the chart

5. **What metric shows memory bandwidth utilization?**
   - [ ] A) sm__throughput
   - [ ] B) dram__throughput
   - [ ] C) launch__grid_size
   - [ ] D) inst_executed

6. **High occupancy guarantees high performance:**
   - [ ] A) True
   - [ ] B) False

7. **What does "achieved occupancy" measure?**
   - [ ] A) Theoretical maximum warps
   - [ ] B) Actual warps resident vs maximum possible
   - [ ] C) Memory usage
   - [ ] D) Compute utilization

8. **To identify bottlenecks, compare:**
   - [ ] A) Achieved vs peak for both memory and compute
   - [ ] B) Only memory metrics
   - [ ] C) Only compute metrics
   - [ ] D) Binary size

9. **Systematic optimization workflow starts with:**
   - [ ] A) Rewriting the kernel
   - [ ] B) Profiling to identify bottleneck
   - [ ] C) Adding more threads
   - [ ] D) Using shared memory

10. **`ncu --set full` does what?**
    - [ ] A) Runs kernel once
    - [ ] B) Collects comprehensive metrics
    - [ ] C) Skips profiling
    - [ ] D) Enables debugging

<details>
<summary>Answers</summary>

1. B - Individual GPU kernels
2. B - Memory bandwidth
3. B - FLOPS / Bytes transferred
4. B - On the sloped (memory) line
5. B - dram__throughput
6. B - False (occupancy is necessary but not sufficient)
7. B - Actual warps resident vs maximum possible
8. A - Achieved vs peak for both memory and compute
9. B - Profiling to identify bottleneck
10. B - Collects comprehensive metrics

</details>

---

## Week 4: Project Templates

### Quiz (10 questions)

1. **A single-file template is best for:**
   - [ ] A) Large projects
   - [ ] B) Quick experiments
   - [ ] C) Production code
   - [ ] D) Multi-GPU

2. **Library templates should expose:**
   - [ ] A) All internal details
   - [ ] B) Clean C++ API hiding CUDA details
   - [ ] C) Raw pointers only
   - [ ] D) No error handling

3. **Benchmark output should include:**
   - [ ] A) Only time
   - [ ] B) Time, bandwidth, FLOPS, and statistics
   - [ ] C) Only FLOPS
   - [ ] D) Source code

4. **Why separate host and device code in libraries?**
   - [ ] A) Faster compilation
   - [ ] B) Users don't need CUDA compiler
   - [ ] C) Required by CUDA
   - [ ] D) Debugging only

5. **A test framework should verify:**
   - [ ] A) Only that code compiles
   - [ ] B) Correctness with known inputs/outputs
   - [ ] C) Performance only
   - [ ] D) Memory usage only

6. **Edge cases in tests should include:**
   - [ ] A) Only typical sizes
   - [ ] B) Zero size, one element, non-power-of-2, large sizes
   - [ ] C) Only powers of 2
    - [ ] D) Only small sizes

7. **CLI applications should support:**
   - [ ] A) Hardcoded parameters only
   - [ ] B) Command-line arguments for configuration
   - [ ] C) No configuration
   - [ ] D) GUI only

8. **Regression testing catches:**
   - [ ] A) New features
   - [ ] B) Performance degradation from changes
   - [ ] C) Compilation errors
   - [ ] D) Memory leaks

9. **A complete template includes:**
   - [ ] A) Just source code
   - [ ] B) Source, build, test, benchmark, documentation
   - [ ] C) Only Makefile
   - [ ] D) Only README

10. **Why use Ninja over Make?**
    - [ ] A) Better error messages
    - [ ] B) Faster parallel builds
    - [ ] C) Required by CUDA
    - [ ] D) Simpler syntax

<details>
<summary>Answers</summary>

1. B - Quick experiments
2. B - Clean C++ API hiding CUDA details
3. B - Time, bandwidth, FLOPS, and statistics
4. B - Users don't need CUDA compiler
5. B - Correctness with known inputs/outputs
6. B - Zero size, one element, non-power-of-2, large sizes
7. B - Command-line arguments for configuration
8. B - Performance degradation from changes
9. B - Source, build, test, benchmark, documentation
10. B - Faster parallel builds

</details>

---

## Scoring

- **9-10 per week:** Ready to proceed
- **7-8 per week:** Review weak areas, then proceed
- **<7 per week:** Re-study the material before Phase 1

**Total Phase 0: 40 questions**
- **36-40:** Excellent foundation
- **32-35:** Good, minor review needed
- **<32:** Spend more time on Phase 0 before continuing
