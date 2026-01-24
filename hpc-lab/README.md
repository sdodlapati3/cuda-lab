# 🖥️ HPC Lab

> **Master HPC workflows for national labs and cluster computing**

This lab develops the HPC workflow skills essential for NESAP, national lab, and scientific computing roles. After completing this module, you'll be able to:

- Write fault-tolerant distributed training scripts
- Navigate HPC job schedulers (Slurm) efficiently
- Build reproducible containerized environments
- Debug multi-node failures systematically

---

## 🎯 NESAP Skill Alignment

| NESAP Requirement | This Lab Covers |
|-------------------|-----------------|
| "NERSC ≠ local workstation" | Slurm job scripts, resource allocation |
| "Can you design a fault-tolerant training workflow?" | Checkpointing, auto-resume |
| "Containers (Docker, Singularity/Apptainer)" | Container building and usage |
| "File systems (Lustre, GPFS)" | I/O optimization for parallel FS |
| "Can you debug a failed multi-node job?" | Distributed debugging techniques |

---

## 📚 Modules

### [01-slurm-basics/](01-slurm-basics/)
**Job Submission & Resource Management**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-submit-monitor | sbatch, squeue, scancel basics | ⭐ |
| ex02-resource-requests | GPU allocation, memory, time limits | ⭐⭐ |
| ex03-job-dependencies | Workflow orchestration with dependencies | ⭐⭐⭐ |
| ex04-job-arrays | Parameter sweeps and hyperparameter tuning | ⭐⭐⭐ |

**Templates provided:**
- `single-gpu-job.sbatch`
- `multi-gpu-job.sbatch`
- `multi-node-job.sbatch`
- `job-array.sbatch`

---

### [02-checkpointing/](02-checkpointing/)
**Fault-Tolerant Training**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-basic-checkpoint | PyTorch model checkpointing | ⭐⭐ |
| ex02-distributed-checkpoint | FSDP/DDP checkpoint strategies | ⭐⭐⭐ |
| ex03-preemption-handling | Graceful shutdown, signal handling | ⭐⭐⭐⭐ |
| ex04-auto-resume | Automatic job resubmission | ⭐⭐⭐⭐ |

**Key skills:** State preservation, preemption recovery, long-running jobs

---

### [03-containers/](03-containers/)
**Singularity/Apptainer for HPC**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-build-container | Definition files, building images | ⭐⭐ |
| ex02-gpu-in-container | NVIDIA runtime, GPU access | ⭐⭐⭐ |
| ex03-mpi-container | MPI + container integration | ⭐⭐⭐⭐ |

**Templates provided:**
- `cuda-pytorch.def` - Base PyTorch + CUDA container
- `build-container.sh` - Build automation

---

### [04-filesystems/](04-filesystems/)
**Parallel Filesystem I/O Optimization**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-io-benchmarking | Measure I/O performance | ⭐⭐ |
| ex02-data-staging | Scratch vs persistent storage | ⭐⭐⭐ |
| ex03-parallel-io | HDF5, parallel writes | ⭐⭐⭐⭐ |

**Key skills:** Lustre/GPFS best practices, avoiding I/O bottlenecks

---

### [05-environment-management/](05-environment-management/)
**Modules, Conda, and Dependencies**

| Content | Description |
|---------|-------------|
| Module system guide | `module load/unload/list` |
| Conda environments | Creating reproducible environments |
| NERSC-specific notes | NERSC module conventions |

---

### [06-debugging-hpc/](06-debugging-hpc/)
**Multi-Node Debugging**

| Exercise | Topic | Difficulty |
|----------|-------|------------|
| ex01-log-analysis | Parsing distributed logs | ⭐⭐ |
| ex02-distributed-debugging | torch.distributed.breakpoint, NCCL_DEBUG | ⭐⭐⭐ |
| ex03-common-failures | OOM, timeout, NCCL errors | ⭐⭐⭐ |

**Reference:** `common-failures.md` - Troubleshooting guide

---

## 🔧 Prerequisites

- Access to HPC cluster with Slurm (or similar)
- Basic Linux command line
- Completed learning-path Weeks 1-12 (or equivalent)
- For multi-GPU exercises: Multi-node allocation capability

---

## 📋 Cluster-Specific Notes

### NERSC (Perlmutter)
```bash
# Load CUDA + PyTorch
module load cudatoolkit pytorch

# GPU allocation
salloc -A <account> -C gpu -q interactive -t 01:00:00 -n 1 --gpus-per-node=1
```

### Generic Slurm Cluster
```bash
# Check available partitions
sinfo

# Check GPU availability
squeue --format="%.10i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

---

## 🎯 Learning Outcomes

After completing this lab, you should be able to:

1. **Write** Slurm job scripts for single-node and multi-node GPU jobs
2. **Implement** fault-tolerant training with automatic checkpoint/resume
3. **Build** Singularity containers for reproducible ML environments
4. **Optimize** I/O patterns for parallel filesystems
5. **Debug** multi-node distributed training failures

---

## 📊 Quick Reference

### Essential Slurm Commands
```bash
sbatch job.sbatch      # Submit job
squeue -u $USER        # Check your jobs
scancel <jobid>        # Cancel job
sinfo                  # Cluster status
sacct -j <jobid>       # Job accounting
```

### Common Environment Variables
```bash
SLURM_JOB_ID           # Current job ID
SLURM_PROCID           # Process rank
SLURM_NNODES           # Number of nodes
SLURM_GPUS_ON_NODE     # GPUs per node
MASTER_ADDR            # For distributed training
MASTER_PORT            # For distributed training
```

### NCCL Debugging
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## 📚 Reference Materials

- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [NERSC Documentation](https://docs.nersc.gov/)
- [Singularity User Guide](https://docs.sylabs.io/guides/latest/user-guide/)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

*Start with [01-slurm-basics/](01-slurm-basics/) to learn job submission, or jump to [02-checkpointing/](02-checkpointing/) if you need fault tolerance immediately.*
