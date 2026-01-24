# HPC Quick Reference

> Essential commands and patterns for GPU clusters and national labs

---

## 🖥️ Slurm Job Scheduler

### Essential Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `sbatch` | Submit batch job | `sbatch job.sbatch` |
| `squeue` | Check job queue | `squeue -u $USER` |
| `scancel` | Cancel job | `scancel 12345` |
| `sinfo` | Cluster status | `sinfo -p gpu` |
| `sacct` | Job accounting | `sacct -j 12345 --format=JobID,State,ExitCode` |
| `salloc` | Interactive allocation | `salloc -n 1 --gpus=1 -t 01:00:00` |
| `srun` | Run within allocation | `srun python train.py` |

### Job Script Template

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=<your_account>

# Load modules
module load cudatoolkit pytorch

# Set distributed training vars
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Run training
srun python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py --config config.yaml
```

### Multi-Node Job

```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

# Get master address from first node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py
```

---

## 📦 Environment Modules

### Common Commands

```bash
module avail              # List available modules
module load <name>        # Load module
module unload <name>      # Unload module
module list               # Show loaded modules
module purge              # Unload all
module show <name>        # Show module details
```

### Typical ML Stack

```bash
module load gcc/11.2.0
module load cuda/12.4
module load cudnn/8.9
module load nccl/2.20
module load python/3.10
module load pytorch/2.1
```

---

## 🐍 Conda Environment Management

### Create Reproducible Environment

```bash
# Create environment
conda create -n cuda-ml python=3.10

# Activate
conda activate cuda-ml

# Install packages
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install flash-attn --no-build-isolation

# Export for reproducibility
conda env export > environment.yml
pip freeze > requirements.txt
```

### Install from File

```bash
conda env create -f environment.yml
pip install -r requirements.txt
```

---

## 💾 Filesystem Best Practices

### NERSC Perlmutter

| Path | Purpose | Quota | Purge Policy |
|------|---------|-------|--------------|
| `$HOME` | Code, configs | 40 GB | None |
| `$SCRATCH` | Large data, checkpoints | 20 TB | 8 weeks |
| `$CFS` | Shared project data | Varies | None |

### I/O Optimization

```python
# Bad: Many small files
for i in range(10000):
    torch.save(data[i], f"data_{i}.pt")

# Good: Single large file or HDF5
torch.save(all_data, "data.pt")

# Better: Memory-mapped for large datasets
import numpy as np
mmap = np.memmap("data.bin", dtype='float32', mode='r', shape=(10000, 1024))
```

### Lustre Striping (for large files)

```bash
# Set striping before creating large files
lfs setstripe -c 32 -S 4M /path/to/directory

# Check current striping
lfs getstripe /path/to/file
```

---

## 🔧 Debugging Distributed Jobs

### NCCL Debugging

```bash
export NCCL_DEBUG=INFO           # Basic info
export NCCL_DEBUG=TRACE          # Detailed trace
export NCCL_DEBUG_SUBSYS=ALL     # All subsystems
```

### PyTorch Distributed Debugging

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
```

### Common NCCL Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `NCCL WARN ... no CUDA device` | Wrong GPU mapping | Check `CUDA_VISIBLE_DEVICES` |
| `NCCL WARN ... timeout` | Network issue or deadlock | Check firewall, increase timeout |
| `NCCL error: unhandled system error` | IB/network config | Check `ibstat`, contact admin |

### Debug Hung Job

```bash
# Find process
squeue -u $USER  # Get node
ssh <node>
nvidia-smi       # Check GPU usage
ps aux | grep python  # Find PIDs

# Attach debugger
py-spy dump --pid <PID>  # Python stack
```

---

## 📊 Performance Monitoring

### nvidia-smi Commands

```bash
nvidia-smi                           # Current status
nvidia-smi -l 1                      # Continuous (1 sec)
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv -l 1
nvidia-smi dmon -s pucvmet -d 1      # Detailed monitoring
```

### GPU Health Check Script

```bash
#!/bin/bash
for i in $(seq 0 $((SLURM_GPUS_ON_NODE-1))); do
    echo "GPU $i:"
    nvidia-smi -i $i --query-gpu=name,temperature.gpu,power.draw,memory.used --format=csv,noheader
done
```

---

## ✅ Checkpointing Pattern

```python
import torch
import os
import signal

class CheckpointHandler:
    def __init__(self, checkpoint_dir, model, optimizer, scheduler=None):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._setup_signal_handler()
    
    def _setup_signal_handler(self):
        """Handle preemption signal."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGUSR1, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, saving checkpoint...")
        self.save("preempt_checkpoint.pt")
        exit(0)
    
    def save(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': self.epoch,
            'global_step': self.global_step,
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return checkpoint['epoch'], checkpoint['global_step']
        return 0, 0
```

---

## 🐳 Container Usage (Singularity/Apptainer)

### Run with GPU

```bash
singularity exec --nv container.sif python train.py
apptainer exec --nv container.sif python train.py
```

### Build Container

```bash
# Definition file: pytorch.def
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:24.01-py3

%post
    pip install flash-attn --no-build-isolation
    pip install wandb tensorboard

%runscript
    python "$@"

# Build
singularity build --fakeroot pytorch.sif pytorch.def
```

---

## 🔗 Useful Links

- [NERSC Documentation](https://docs.nersc.gov/)
- [Slurm Documentation](https://slurm.schedmd.com/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)

---

*Keep this reference handy when working on HPC clusters!*
