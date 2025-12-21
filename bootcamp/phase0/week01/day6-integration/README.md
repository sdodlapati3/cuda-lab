# Day 6: Integration + Project Template

## Goals Today

1. Review and consolidate Week 1 learnings
2. Create a reusable project template
3. Set up your lab notebook structure
4. Push everything to your repositories

## The Project Template

After this week, you should have a template for all future CUDA projects:

```
my_cuda_project/
├── CMakeLists.txt          # Modern CMake with multi-arch support
├── build.sh                # Quick build script
├── include/
│   ├── cuda_timer.cuh      # Event-based timing
│   ├── benchmark.cuh       # Benchmarking framework
│   └── cuda_utils.cuh      # Error checking, device info
├── src/
│   └── main.cu
├── kernels/
│   └── my_kernel.cu
├── tests/
│   └── test_correctness.cu
├── scripts/
│   ├── profile.sh          # Nsight Compute wrapper
│   └── extract_sass.sh     # SASS analysis
└── README.md
```

## Week 1 Checklist

Before moving to Week 2, verify:

- [ ] **CMake works:** Can build with `cmake -G Ninja -B build && ninja -C build`
- [ ] **Understand flags:** Know what `-O3`, `-lineinfo`, `-arch=sm_XX` do
- [ ] **Can read SASS:** Extracted and annotated at least one kernel
- [ ] **Benchmark harness:** Have reusable timing code
- [ ] **Roofline:** Know your GPU's peak BW and compute, plotted roofline

## Lab Notebook Entry

Create your first lab notebook entry:

```markdown
# Week 1: Build System Mastery

## Date: YYYY-MM-DD

## What I Learned
- CMake treats CUDA as first-class language since 3.18
- `-lineinfo` preserves source mapping without disabling opts
- PTX is portable IR, SASS is actual GPU assembly
- My GPU: [name], Peak BW: [X] GB/s, Peak FP32: [Y] GFLOPS

## Key Insights
- Most kernels are memory-bound (AI < ridge point)
- Always use CUDA events for timing, not wall clock
- The `-G` flag is for debugging only (kills performance)

## Measurements
| Kernel | AI | Achieved | % of Roofline |
|--------|----|---------:|-------------:|
| vector_add | 0.083 | XX GB/s | XX% |
| ... | ... | ... | ... |

## Questions for Week 2
- How does occupancy affect performance?
- When should I use shared memory?
```

## Setting Up Your Repositories

### 1. Lab Notebook Repository

```bash
mkdir -p ~/cuda-lab-notebook
cd ~/cuda-lab-notebook
git init

mkdir -p daily microbench insights
echo "# CUDA Lab Notebook" > README.md
echo "Daily journal of GPU programming journey." >> README.md

git add .
git commit -m "Initial lab notebook setup"
```

### 2. Kernel Zoo Repository

```bash
mkdir -p ~/kernel-zoo
cd ~/kernel-zoo
git init

mkdir -p primitives/{reduction,scan,transpose} gemm ml_ops benchmarks
echo "# Kernel Zoo" > README.md
echo "Collection of optimized CUDA kernels." >> README.md

git add .
git commit -m "Initial kernel zoo setup"
```

## Exercise: Create Your First Kernel Zoo Entry

Take the reduction kernel from Day 4 and create a proper kernel zoo entry:

```
kernel-zoo/primitives/reduction/
├── naive.cu              # First version
├── warp_shuffle.cu       # Optimized version
├── benchmark.py          # Python wrapper for timing
├── test_correctness.py   # Verify against numpy
└── README.md             # Document optimization journey
```

## Next Week Preview

**Week 2: Debugging Foundations**
- compute-sanitizer for race detection
- cuda-gdb for breakpoint debugging
- Nsight Systems for timeline analysis
- Understanding async error handling

You'll need the build system mastery from this week to effectively debug!
