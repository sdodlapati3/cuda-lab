# Week 2: Debugging Foundations

> **Goal:** Find and fix bugs in CUDA code with confidence using professional debugging tools.

## Daily Schedule

| Day | Topic | Deliverable |
|-----|-------|-------------|
| 1 | compute-sanitizer | Race condition detection |
| 2 | Memory error detection | Out-of-bounds, leaks |
| 3 | cuda-gdb basics | Breakpoint debugging |
| 4 | Async error handling | Proper error propagation |
| 5 | Nsight Systems | Timeline analysis |
| 6 | Integration | Debug workflow template |

## Prerequisites

```bash
# Check your tools
compute-sanitizer --version  # Memory/race checker
cuda-gdb --version           # CUDA debugger
nsys --version               # Nsight Systems profiler
```

## Quick Start

```bash
# Day 1: Find race conditions
cd day1-compute-sanitizer
./build.sh
compute-sanitizer --tool racecheck ./race_example
```

## Week 2 Checklist

- [ ] Can detect race conditions with compute-sanitizer
- [ ] Can find memory errors (OOB, uninitialized, leaks)
- [ ] Can set breakpoints and inspect GPU state in cuda-gdb
- [ ] Understand async error handling patterns
- [ ] Can analyze GPU timeline with Nsight Systems
- [ ] Have a systematic debugging workflow

## Key Insight

> **GPU debugging is different from CPU debugging.**
> 
> - Thousands of threads running simultaneously
> - Errors may appear non-deterministic
> - printf debugging is limited
> - Many errors are silent (wrong results, not crashes)

## Resources

- [CUDA-MEMCHECK (compute-sanitizer)](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [CUDA-GDB User Guide](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
