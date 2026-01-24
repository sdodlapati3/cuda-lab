# Slurm Basics

> **Master HPC job submission and resource management**

Learn to use Slurm, the most common HPC job scheduler, for GPU workloads.

---

## 🎯 Learning Objectives

- Submit batch jobs with proper GPU allocation
- Monitor and manage job queues
- Use job arrays for parameter sweeps
- Create multi-node GPU job scripts

---

## 📚 Exercises

| Exercise | Topic | Time | Difficulty |
|----------|-------|------|------------|
| [ex01-submit-monitor](ex01-submit-monitor/) | sbatch, squeue, scancel | 30 min | ⭐ |
| [ex02-resource-requests](ex02-resource-requests/) | GPU allocation, memory | 45 min | ⭐⭐ |
| [ex03-job-dependencies](ex03-job-dependencies/) | Workflow orchestration | 1 hr | ⭐⭐⭐ |
| [ex04-job-arrays](ex04-job-arrays/) | Parameter sweeps | 1 hr | ⭐⭐⭐ |

---

## 📁 Templates

Ready-to-use job script templates:

| Template | Use Case |
|----------|----------|
| [single-gpu-job.sbatch](templates/single-gpu-job.sbatch) | Single GPU training/inference |
| [multi-gpu-job.sbatch](templates/multi-gpu-job.sbatch) | Multi-GPU single node (DDP) |
| [multi-node-job.sbatch](templates/multi-node-job.sbatch) | Multi-node distributed training |
| [job-array.sbatch](templates/job-array.sbatch) | Hyperparameter sweeps |
| [interactive.sh](templates/interactive.sh) | Interactive GPU session |

---

## 🔧 Quick Reference

### Essential Commands

```bash
# Submit a job
sbatch job.sbatch

# Check your jobs
squeue -u $USER

# Check all jobs on partition
squeue -p gpu

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Get detailed job info
scontrol show job <job_id>

# Check job accounting (after completion)
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,ExitCode
```

### Key SBATCH Directives

```bash
#SBATCH --job-name=train           # Job name
#SBATCH --output=logs/%j.out       # Stdout (%j = job ID)
#SBATCH --error=logs/%j.err        # Stderr
#SBATCH --time=24:00:00            # Wall time limit
#SBATCH --partition=gpu            # Partition name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=4        # Tasks (usually = GPUs)
#SBATCH --gpus-per-node=4          # GPUs per node
#SBATCH --cpus-per-task=8          # CPUs per task
#SBATCH --mem=64G                  # Memory per node
#SBATCH --account=<account>        # Allocation account
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SLURM_JOB_ID` | Current job ID |
| `SLURM_JOB_NAME` | Job name |
| `SLURM_NNODES` | Number of nodes |
| `SLURM_NTASKS` | Total number of tasks |
| `SLURM_PROCID` | Task rank (0 to NTASKS-1) |
| `SLURM_LOCALID` | Local task ID within node |
| `SLURM_NODELIST` | List of allocated nodes |
| `SLURM_GPUS_ON_NODE` | GPUs on this node |

---

## 📖 Further Reading

- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [NERSC Job Script Generator](https://my.nersc.gov/script_generator.php)
