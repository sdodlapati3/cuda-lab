# Phase 0: Professional Development Environment

> **Duration:** 4 weeks
> **Goal:** Build, run, profile, and debug CUDA with confidence before writing real kernels.

## Why Phase 0?

Most CUDA tutorials jump straight into kernels. This fails because:
- You can't measure performance without profiling infrastructure
- You can't trust results without debugging skills
- You can't iterate fast without proper build systems

**Phase 0 makes you dangerous** before you write a single production kernel.

---

## Essential Resources

### 📚 Daily Reference Spine
**[Read this first →](daily-reference-spine.md)**

The CUDA Programming Guide and Best Practices Guide should be open every day. This document maps each phase to specific sections.

### 🔧 Library-First Guide  
**[Read this second →](library-first-guide.md)**

Before writing custom kernels, know when to use cuBLAS, cuDNN, CUB, CUTLASS, or Triton. The best kernel is one you don't have to write.

---

## Weekly Schedule

| Week | Topic | Deliverables |
|------|-------|--------------|
| [Week 1](week01/) | Build System Mastery | CMake project template, PTX/SASS analysis |
| [Week 2](week02/) | Debugging Foundations | Debug workflow, compute-sanitizer + cuda-gdb skills |
| [Week 3](week03/) | Performance Analysis | Nsight Compute profiling, roofline thinking |
| [Week 4](week04/) | Project Templates | 6 production-ready project templates |

---

## Phase 0 Checklist

### Build System (Week 1)
- [ ] Can configure CMake for multi-arch CUDA builds
- [ ] Understand `-O3`, `-lineinfo`, `-arch=sm_XX` flags
- [ ] Can inspect PTX and SASS output
- [ ] Have reproducible build scripts

### Debugging (Week 2)
- [ ] Can detect race conditions with compute-sanitizer
- [ ] Can find memory errors (OOB, uninitialized, leaks)
- [ ] Can use cuda-gdb for breakpoint debugging
- [ ] Understand async error handling patterns
- [ ] Can analyze GPU timeline with Nsight Systems

### Performance Analysis (Week 3)
- [ ] Can profile with Nsight Compute (ncu)
- [ ] Understand memory vs compute bound kernels
- [ ] Can read roofline charts
- [ ] Can identify bottlenecks from metrics
- [ ] Have systematic optimization workflow

### Project Templates (Week 4)
- [ ] Have single-file quick-start template
- [ ] Have library template with clean API
- [ ] Have application template with CLI
- [ ] Have benchmark template with CSV/JSON output
- [ ] Have test framework with assertions
- [ ] Have complete all-in-one template

---

## Key Principle: Profiling as Reflex

> **Suggestion from curriculum review:** Introduce Nsight earlier (Week 2-3), even on toy kernels, so profiling becomes reflex—not a later "phase."

We've implemented this:
- **Week 2, Day 5:** Nsight Systems (timeline analysis)
- **Week 3, All Days:** Nsight Compute (kernel metrics)

**By the end of Phase 0, you should automatically profile any kernel you write.**

---

## Key Principle: Library-First Development

> **Suggestion from curriculum review:** Add one explicit "library-first" checkpoint: learn when NOT to write kernels because cuBLAS/cuDNN/CUTLASS/Inductor already solves it.

See [library-first-guide.md](library-first-guide.md) for the complete decision framework.

**Rule of thumb:**
1. Can cuBLAS/cuDNN do it? → Use them
2. Can CUB do it? → Use it
3. Can Triton do it 90% as fast? → Use Triton
4. Only then → Write custom CUDA

---

## Key Principle: Official Docs as Daily Reference

> **Suggestion from curriculum review:** Make the CUDA Programming Guide + Best Practices Guide your daily reference spine (not optional reading).

See [daily-reference-spine.md](daily-reference-spine.md) for section-by-section mapping.

**Daily practice:**
1. Before coding, read the relevant section
2. While coding, have the docs open
3. After coding, re-read to fill gaps

---

## Tools Required

```bash
# Verify your installation
nvcc --version                    # CUDA compiler
cmake --version                   # CMake 3.18+
ninja --version                   # Ninja build system (optional but recommended)
compute-sanitizer --version       # Memory/race checker
cuda-gdb --version                # CUDA debugger
nsys --version                    # Nsight Systems
ncu --version                     # Nsight Compute
```

---

## Quick Start

```bash
# Clone and enter
cd cuda-lab/bootcamp/phase0

# Week 1: Build systems
cd week01/day1-simple-cmake
./build.sh
./build/hello

# Week 2: Debugging
cd ../week02/day1-compute-sanitizer
./build.sh
compute-sanitizer --tool racecheck ./build/race_example

# Week 3: Performance
cd ../week03/day1-ncu-basics
./build.sh
ncu --set full ./build/vector_ops

# Week 4: Templates
cd ../week04/day6-complete
./build.sh
./build/myapp
./build/tests
./build/benchmarks
```

---

## Phase 0 Complete → Ready for Phase 1

After Phase 0, you have:
1. **Build infrastructure** that just works
2. **Debugging reflexes** to catch bugs fast
3. **Profiling discipline** to measure everything
4. **Project templates** to start any new work quickly
5. **Library awareness** to avoid reinventing wheels

**Next:** [Phase 1: CUDA Fundamentals](../README.md#phase-1-cuda-fundamentals-weeks-3-6)
