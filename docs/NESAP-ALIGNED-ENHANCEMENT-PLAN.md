# CUDA-Lab Enhancement Plan: NESAP/HPC Career Alignment

> **Created:** January 24, 2026  
> **Goal:** Transform cuda-lab into a comprehensive portfolio that demonstrates NESAP-ready skills  
> **Target Role:** NERSC/NESAP ML Postdoc (Systems-aware ML at HPC scale)

---

## 📋 Executive Summary

This plan addresses three objectives:
1. **Reorganize** - Eliminate redundant folders, consolidate learning paths
2. **Expand** - Add missing HPC/scientific computing content
3. **Align** - Map all content to NESAP skill requirements

### Current State Assessment

| Folder | Purpose | Redundancy Issue | Action |
|--------|---------|------------------|--------|
| `learning-path/` | 18-week interactive notebooks | ✅ Core curriculum | **KEEP - Primary** |
| `bootcamp/` | 52-week intensive curriculum | ✅ Advanced track | **KEEP - Advanced** |
| `tutorials/` | Markdown tutorials | ❌ Duplicates learning-path, 1 file exists | **DEPRECATE** |
| `notes/` | Personal notes, curriculum drafts | ⚠️ `cuda-learning-curriculum.md` duplicates tutorials | **CONSOLIDATE** |
| `cuda-programming-guide/` | Reference documentation | ✅ Unique reference | **KEEP** |
| `practice/` | Hands-on exercises | ✅ Complements learning-path | **KEEP + EXPAND** |
| `blog-templates/` | Blog conversion templates | ⚠️ Empty infrastructure | **DEFER** |

---

## 🗂️ Part 1: Organizational Cleanup

### 1.1 Deprecate `tutorials/` Folder

**Problem:** The `tutorials/` folder promises 40+ tutorials but contains only 1 file. All this content already exists in better form in `learning-path/` notebooks.

**Action:**
```bash
# Move the single existing tutorial to notes/ as reference
mv tutorials/01-foundations/01-cpu-vs-gpu.md notes/reference/
# Remove empty tutorials structure
rm -rf tutorials/
```

**Update `mkdocs.yml`** to remove tutorials from navigation.

---

### 1.2 Consolidate `notes/` Folder

**Problem:** `notes/cuda-learning-curriculum.md` (566 lines) duplicates tutorial planning that's already implemented in `learning-path/`.

**Action:**
```
notes/                          # BEFORE
├── cuda-learning-curriculum.md   # Redundant planning doc
└── cuda-quick-reference.md       # Useful cheatsheet

notes/                          # AFTER
├── cuda-quick-reference.md       # Keep
├── reference/                    # New
│   └── cpu-vs-gpu.md             # From tutorials/
└── archive/                      # Archive old planning
    └── cuda-learning-curriculum.md
```

---

### 1.3 Clarify Learning Path vs Bootcamp Relationship

**Problem:** Unclear when to use 18-week learning-path vs 52-week bootcamp.

**Solution:** Add a top-level `LEARNING-TRACKS.md`:

```markdown
# cuda-lab Learning Tracks

## Track 1: Foundation (learning-path/)
- **Duration:** 18 weeks part-time
- **Audience:** Anyone learning CUDA
- **Outcome:** Working CUDA proficiency

## Track 2: Mastery (bootcamp/)  
- **Duration:** 52 weeks full-time
- **Audience:** ML engineers targeting performance roles
- **Prerequisites:** Complete Track 1 or equivalent
- **Outcome:** Expert-level GPU performance engineering

## Track 3: NESAP Preparation (NEW)
- **Duration:** 12 weeks intensive
- **Audience:** Targeting HPC/scientific ML roles
- **Focus:** Distributed training, profiling, HPC workflows
```

---

## 🎯 Part 2: NESAP Skill Gap Analysis

### Mapping NESAP Requirements to Current Content

| NESAP Skill | Current Coverage | Gap Level | Priority |
|-------------|------------------|-----------|----------|
| **Deep Learning Fundamentals** | ❌ Assumed prerequisite | HIGH | Add ML foundations module |
| **Scientific ML (PINNs, UQ)** | ❌ Not covered | HIGH | New module needed |
| **Distributed Training (DDP, FSDP)** | ✅ bootcamp/phase8 | LOW | Expand with benchmarks |
| **GPU Architecture Awareness** | ✅ Excellent coverage | NONE | Complete |
| **Performance Profiling** | ⚠️ Partial (theory, light practice) | MEDIUM | Add profiling lab |
| **Python at Scale** | ❌ CUDA C++ focus | MEDIUM | Add PyTorch optimization |
| **CUDA/Triton/C++ Extensions** | ✅ bootcamp/phase8 | LOW | Complete |
| **Large-Scale Data Pipelines** | ❌ Not covered | HIGH | New module needed |
| **HPC Workflows (Slurm, containers)** | ⚠️ Mentioned, not practiced | HIGH | Add HPC lab |
| **Linux & HPC Environments** | ❌ Assumed | MEDIUM | Add quick reference |
| **Benchmarking & Scaling Metrics** | ⚠️ Concepts only | HIGH | Add benchmark suite |
| **Scientific Domains** | ❌ Not covered | MEDIUM | Add case studies |

---

## 🔧 Part 3: New Content Modules

### 3.1 Performance Profiling Lab (HIGH PRIORITY)

**Location:** `profiling-lab/` (new top-level directory)

**Why NESAP cares:** "Performance bugs often come from hardware misunderstandings... NESAP success = measured improvements, not guesses."

```
profiling-lab/
├── README.md                         # Profiling philosophy & tools overview
├── 01-nsight-systems/
│   ├── README.md                     # Timeline analysis, CPU-GPU overlap
│   ├── exercises/
│   │   ├── ex01-timeline-basics/     # Read timeline, identify stalls
│   │   ├── ex02-kernel-overlap/      # Streams and concurrency
│   │   ├── ex03-memory-timeline/     # H2D/D2H transfer analysis
│   │   └── ex04-multi-gpu-timeline/  # NCCL communication profiling
│   └── case-studies/
│       ├── case01-transformer-training.md
│       └── case02-inference-latency.md
├── 02-nsight-compute/
│   ├── README.md                     # Kernel-level profiling
│   ├── exercises/
│   │   ├── ex01-memory-metrics/      # Bandwidth, cache hit rates
│   │   ├── ex02-compute-metrics/     # Occupancy, warp efficiency
│   │   ├── ex03-roofline-practice/   # Plot kernels, identify bottlenecks
│   │   └── ex04-optimization-loop/   # Profile → optimize → reprofile
│   └── reference/
│       ├── key-metrics-cheatsheet.md # Most important metrics explained
│       └── common-bottlenecks.md     # Memory-bound vs compute-bound
├── 03-pytorch-profiler/
│   ├── README.md                     # PyTorch Profiler + TensorBoard
│   ├── exercises/
│   │   ├── ex01-basic-profiling/     # torch.profiler basics
│   │   ├── ex02-memory-profiling/    # Memory snapshots
│   │   └── ex03-distributed-profiling/ # DDP profiling
│   └── templates/
│       └── profiling-harness.py      # Reusable profiling wrapper
├── 04-energy-profiling/              # NESAP: "science per joule"
│   ├── README.md                     # nvidia-smi, NVML, power monitoring
│   ├── exercises/
│   │   ├── ex01-power-measurement/
│   │   └── ex02-energy-efficiency/
│   └── scripts/
│       └── energy-benchmark.py
└── 05-scaling-benchmarks/
    ├── README.md                     # Strong vs weak scaling
    ├── exercises/
    │   ├── ex01-single-gpu-baseline/
    │   ├── ex02-multi-gpu-scaling/
    │   └── ex03-communication-overhead/
    └── templates/
        ├── scaling-benchmark-template.py
        └── scaling-report-template.md
```

**Key Deliverables:**
- [ ] Nsight Systems timeline analysis for DDP training
- [ ] Nsight Compute optimization loop (3 iterations minimum)
- [ ] Roofline plot of reduction/matmul kernels with annotations
- [ ] Energy efficiency benchmark comparing implementations
- [ ] Scaling efficiency report (90%+ efficiency at 4 GPUs)

---

### 3.2 HPC Workflows Lab (HIGH PRIORITY)

**Location:** `hpc-lab/` (new top-level directory)

**Why NESAP cares:** "NERSC ≠ local workstation... Can you design a fault-tolerant training workflow?"

```
hpc-lab/
├── README.md                         # HPC mindset for ML practitioners
├── 01-slurm-basics/
│   ├── README.md                     # Job submission, arrays, dependencies
│   ├── templates/
│   │   ├── single-gpu-job.sbatch
│   │   ├── multi-gpu-job.sbatch
│   │   ├── multi-node-job.sbatch
│   │   └── job-array.sbatch
│   └── exercises/
│       ├── ex01-submit-monitor/      # sbatch, squeue, scancel
│       ├── ex02-resource-requests/   # GPU allocation, memory
│       └── ex03-job-dependencies/    # Workflow orchestration
├── 02-checkpointing/
│   ├── README.md                     # Fault-tolerant training
│   ├── examples/
│   │   ├── pytorch-checkpoint.py
│   │   ├── distributed-checkpoint.py
│   │   └── auto-resume.sbatch
│   └── exercises/
│       ├── ex01-basic-checkpoint/
│       ├── ex02-distributed-checkpoint/
│       └── ex03-preemption-handling/
├── 03-containers/
│   ├── README.md                     # Singularity/Apptainer for HPC
│   ├── templates/
│   │   ├── cuda-pytorch.def          # Container definition
│   │   └── build-container.sh
│   └── exercises/
│       ├── ex01-build-container/
│       ├── ex02-gpu-in-container/
│       └── ex03-mpi-container/
├── 04-filesystems/
│   ├── README.md                     # Lustre, GPFS, scratch vs home
│   ├── best-practices.md             # I/O patterns for HPC
│   └── exercises/
│       ├── ex01-io-benchmarking/
│       └── ex02-data-staging/
├── 05-environment-management/
│   ├── README.md                     # Modules, conda, pip
│   ├── templates/
│   │   ├── environment.yml
│   │   └── setup-env.sh
│   └── nersc-specific.md             # NERSC module system
└── 06-debugging-hpc/
    ├── README.md                     # Debugging multi-node jobs
    ├── common-failures.md            # Timeout, OOM, NCCL errors
    └── exercises/
        ├── ex01-log-analysis/
        └── ex02-distributed-debugging/
```

**Key Deliverables:**
- [ ] Fault-tolerant training script with auto-resume
- [ ] Multi-node job script with proper resource allocation
- [ ] Singularity container for reproducible ML environment
- [ ] I/O benchmark showing optimal data loading patterns

---

### 3.3 Scientific ML Module (HIGH PRIORITY)

**Location:** `learning-path/scientific-ml/` OR `bootcamp/scientific-ml/`

**Why NESAP cares:** "NESAP projects almost always couple ML to simulations or scientific pipelines."

```
scientific-ml/
├── README.md                         # Scientific ML overview
├── 01-pinns/
│   ├── README.md                     # Physics-Informed Neural Networks
│   ├── theory/
│   │   ├── pinn-formulation.md
│   │   └── loss-function-design.md
│   ├── examples/
│   │   ├── 1d-heat-equation.ipynb
│   │   ├── burgers-equation.ipynb
│   │   └── navier-stokes-2d.ipynb
│   └── exercises/
│       └── ex01-custom-pinn/
├── 02-surrogate-models/
│   ├── README.md                     # Replacing expensive simulations
│   ├── examples/
│   │   ├── neural-operator.ipynb
│   │   └── autoencoder-dynamics.ipynb
│   └── exercises/
│       └── ex01-simulation-surrogate/
├── 03-hybrid-solvers/
│   ├── README.md                     # ML + numerical methods
│   ├── examples/
│   │   ├── ml-preconditioner.ipynb
│   │   └── learned-correction.ipynb
│   └── exercises/
│       └── ex01-hybrid-system/
├── 04-uncertainty-quantification/
│   ├── README.md                     # UQ methods
│   ├── examples/
│   │   ├── mc-dropout.ipynb
│   │   ├── deep-ensembles.ipynb
│   │   └── bayesian-nn.ipynb
│   └── exercises/
│       └── ex01-uq-pipeline/
└── 05-case-studies/
    ├── climate-emulator.md
    ├── materials-property-prediction.md
    └── particle-physics-reconstruction.md
```

---

### 3.4 Data Pipeline Module (HIGH PRIORITY)

**Location:** `learning-path/data-pipelines/` or integrate into existing weeks

**Why NESAP cares:** "I/O often dominates ML at scale... Can you explain how to avoid GPU starvation due to I/O?"

```
data-pipelines/
├── README.md
├── 01-parallel-loading/
│   ├── README.md
│   ├── pytorch-dataloader.ipynb      # num_workers, pin_memory
│   ├── prefetching-patterns.ipynb
│   └── exercises/
├── 02-large-datasets/
│   ├── README.md
│   ├── sharding-strategies.ipynb
│   ├── memory-mapped-datasets.ipynb
│   ├── streaming-datasets.ipynb
│   └── exercises/
├── 03-io-optimization/
│   ├── README.md
│   ├── hdf5-patterns.ipynb
│   ├── webdataset.ipynb
│   └── exercises/
└── 04-distributed-data/
    ├── README.md
    ├── distributed-sampler.ipynb
    └── exercises/
```

---

### 3.5 Benchmarking Suite (HIGH PRIORITY)

**Location:** `benchmarks/` (new top-level directory)

**Why NESAP cares:** "Scaling efficiency, time-to-solution, energy efficiency... Can you design a fair scaling benchmark?"

```
benchmarks/
├── README.md                         # Benchmarking philosophy
├── kernels/
│   ├── reduction/
│   │   ├── benchmark.py
│   │   ├── baselines/                # cuBLAS, CUB reference
│   │   └── results/                  # CSV/JSON results
│   ├── matmul/
│   ├── softmax/
│   └── attention/
├── scaling/
│   ├── strong-scaling/
│   │   ├── benchmark.py
│   │   └── plot-scaling.py
│   ├── weak-scaling/
│   └── communication-overhead/
├── hardware-baselines/
│   ├── T4.json                       # Baseline numbers for comparison
│   ├── A100-40GB.json
│   ├── A100-80GB.json
│   └── H100.json
├── roofline/
│   ├── generate-roofline.py
│   ├── plot-roofline.py
│   └── reference-plots/
├── energy/
│   ├── power-benchmark.py
│   └── efficiency-report.py
└── templates/
    ├── benchmark-template.py
    ├── scaling-report-template.md
    └── regression-test.py
```

---

## 🔄 Part 4: Content Updates to Existing Modules

### 4.1 Enhance `practice/06-systems/` 

Add missing exercises:

```
practice/06-systems/                  # CURRENT
├── ex01-ipc-producer-consumer/
├── ex02-texture-image-processing/

practice/06-systems/                  # ENHANCED
├── ex01-ipc-producer-consumer/
├── ex02-texture-image-processing/
├── ex03-production-error-handling/   # NEW: Async errors, watchdogs
├── ex04-gpu-health-monitoring/       # NEW: NVML, health checks
├── ex05-multi-process-inference/     # NEW: Triton-style serving
└── ex06-mig-partitioning/            # NEW: A100/H100 MIG
```

---

### 4.2 Add PyTorch Optimization Track to Bootcamp

**Location:** `bootcamp/pytorch-optimization/` (supplement to phase8)

```
pytorch-optimization/
├── README.md
├── 01-torch-compile/
│   ├── basics.ipynb
│   ├── debugging-failures.ipynb
│   └── custom-backends.ipynb
├── 02-memory-optimization/
│   ├── gradient-checkpointing.ipynb
│   ├── activation-checkpointing.ipynb
│   └── memory-efficient-attention.ipynb
├── 03-distributed-optimization/
│   ├── ddp-tuning.ipynb
│   ├── fsdp-sharding.ipynb
│   └── pipeline-parallelism.ipynb
└── 04-inference-optimization/
    ├── export-optimization.ipynb
    ├── quantization.ipynb
    └── batching-strategies.ipynb
```

---

### 4.3 Add Quick Reference for HPC/Linux

**Location:** `notes/hpc-quick-reference.md`

```markdown
# HPC Quick Reference

## Slurm Commands
- sbatch, squeue, scancel, sinfo, sacct

## Environment Modules  
- module load/unload/list/avail

## Common NERSC Modules
- python, pytorch, cuda, cudnn, nccl

## Filesystem Layout
- $HOME (small, backed up)
- $SCRATCH (large, not backed up)
- $CFS (community shared)

## Debugging Distributed Jobs
- NCCL_DEBUG=INFO
- CUDA_LAUNCH_BLOCKING=1
- torch.distributed.breakpoint()
```

---

## 📅 Part 5: Implementation Timeline

### Phase 1: Cleanup (Week 1)
- [ ] Deprecate `tutorials/` folder
- [ ] Consolidate `notes/` folder
- [ ] Create `LEARNING-TRACKS.md`
- [ ] Update main `README.md` navigation

### Phase 2: Profiling Lab (Weeks 2-3)
- [ ] Create `profiling-lab/` structure
- [ ] Nsight Systems exercises (4)
- [ ] Nsight Compute exercises (4)
- [ ] PyTorch profiler exercises (3)
- [ ] Energy profiling module
- [ ] Scaling benchmarks template

### Phase 3: HPC Lab (Weeks 4-5)
- [ ] Create `hpc-lab/` structure
- [ ] Slurm templates and exercises
- [ ] Checkpointing module
- [ ] Container templates
- [ ] Filesystem best practices

### Phase 4: Scientific ML (Weeks 6-8)
- [ ] PINN examples and exercises
- [ ] Surrogate model module
- [ ] UQ methods module
- [ ] Case studies (3)

### Phase 5: Benchmarks & Polish (Weeks 9-10)
- [ ] Benchmark suite setup
- [ ] Hardware baselines
- [ ] Roofline generation tools
- [ ] Scaling benchmark templates

### Phase 6: Integration (Weeks 11-12)
- [ ] Data pipelines module
- [ ] PyTorch optimization track
- [ ] Cross-link all modules
- [ ] Final documentation pass

---

## 🎯 Part 6: NESAP Readiness Checklist

After completing this enhancement plan, verify:

### Systems-Aware ML Thinking
- [ ] Can explain memory-bound vs compute-bound with profiler evidence
- [ ] Can diagnose scaling efficiency drops
- [ ] Can identify when Python abstraction is the bottleneck

### Performance Modeling
- [ ] Can plot kernels on roofline and explain position
- [ ] Can predict performance improvement from optimization
- [ ] Can design fair scaling benchmarks

### ML + Simulation Integration
- [ ] Have implemented PINN or surrogate model
- [ ] Understand hybrid solver patterns
- [ ] Can apply UQ methods

### Efficiency Narratives
- [ ] Can write optimization report with before/after metrics
- [ ] Can present scaling results clearly
- [ ] Have documented case studies in portfolio

---

## 📊 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Profiling exercises completed | 15+ | Checklist |
| HPC workflow exercises | 10+ | Checklist |
| Scientific ML examples | 5+ | Notebook count |
| Benchmark baselines | 4 GPUs | JSON files |
| Scaling efficiency demonstrated | >90% at 4 GPU | Benchmark results |
| Portfolio case studies | 3+ | Markdown docs |
| NESAP skill coverage | >90% | Gap analysis |

---

## 🔗 Related Documents

- [CURRICULUM-ENHANCEMENT-PLAN.md](CURRICULUM-ENHANCEMENT-PLAN.md) - Previous enhancement (completed)
- [ADVANCED-TOPICS-ENHANCEMENT.md](ADVANCED-TOPICS-ENHANCEMENT.md) - Advanced CUDA features
- [modern-gpu-ecosystem.md](modern-gpu-ecosystem.md) - Tool selection guide
- [notebook-quality-guide.md](notebook-quality-guide.md) - Notebook standards

---

## Appendix A: Proposed Final Directory Structure

```
cuda-lab/
├── README.md                         # Main entry point
├── LEARNING-TRACKS.md                # NEW: Track overview
├── mkdocs.yml
│
├── learning-path/                    # 18-week foundation curriculum
│   ├── README.md
│   ├── week-01/ ... week-18/
│   ├── scientific-ml/                # NEW: PINN, surrogate, UQ
│   └── data-pipelines/               # NEW: I/O optimization
│
├── bootcamp/                         # 52-week mastery curriculum
│   ├── README.md
│   ├── phase0/ ... phase8/
│   ├── pytorch-optimization/         # NEW: torch.compile, DDP tuning
│   ├── starters/
│   ├── templates/
│   └── capstones/
│
├── profiling-lab/                    # NEW: Performance analysis
│   ├── 01-nsight-systems/
│   ├── 02-nsight-compute/
│   ├── 03-pytorch-profiler/
│   ├── 04-energy-profiling/
│   └── 05-scaling-benchmarks/
│
├── hpc-lab/                          # NEW: HPC workflows
│   ├── 01-slurm-basics/
│   ├── 02-checkpointing/
│   ├── 03-containers/
│   ├── 04-filesystems/
│   └── 05-debugging-hpc/
│
├── benchmarks/                       # NEW: Benchmark suite
│   ├── kernels/
│   ├── scaling/
│   ├── hardware-baselines/
│   └── roofline/
│
├── cuda-programming-guide/           # Reference documentation
│   └── (unchanged)
│
├── practice/                         # Hands-on exercises
│   ├── 01-foundations/
│   ├── 02-memory/
│   ├── 03-parallel/
│   ├── 04-optimization/
│   ├── 05-advanced/
│   └── 06-systems/                   # ENHANCED
│
├── notes/                            # CONSOLIDATED
│   ├── cuda-quick-reference.md
│   ├── hpc-quick-reference.md        # NEW
│   ├── reference/
│   └── archive/
│
├── docs/                             # Planning & guides
│   ├── NESAP-ALIGNED-ENHANCEMENT-PLAN.md  # THIS DOCUMENT
│   ├── CURRICULUM-ENHANCEMENT-PLAN.md
│   ├── ADVANCED-TOPICS-ENHANCEMENT.md
│   └── modern-gpu-ecosystem.md
│
├── blog-templates/                   # DEFERRED
│
└── scripts/                          # Utility scripts
    └── (unchanged)
```

---

*Last updated: January 24, 2026*
