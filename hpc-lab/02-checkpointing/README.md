# Checkpointing - Fault-Tolerant Training

> **Build resilient training pipelines that survive preemption and failures**

HPC jobs can be preempted, hit wall time limits, or fail unexpectedly. Proper checkpointing ensures you never lose progress.

---

## 🎯 Learning Objectives

- Implement robust checkpoint save/load
- Handle preemption signals gracefully
- Set up automatic job resubmission
- Use distributed checkpointing for multi-GPU

---

## 📚 Exercises

| Exercise | Topic | Time | Difficulty |
|----------|-------|------|------------|
| [ex01-basic-checkpoint](ex01-basic-checkpoint/) | PyTorch model checkpointing | 1 hr | ⭐⭐ |
| [ex02-distributed-checkpoint](ex02-distributed-checkpoint/) | DDP/FSDP checkpoints | 1.5 hr | ⭐⭐⭐ |
| [ex03-preemption-handling](ex03-preemption-handling/) | Signal handling, graceful shutdown | 1 hr | ⭐⭐⭐⭐ |
| [ex04-auto-resume](ex04-auto-resume/) | Automatic job resubmission | 1 hr | ⭐⭐⭐⭐ |

---

## 💡 Key Concepts

### What to Checkpoint

| Component | Why |
|-----------|-----|
| Model state dict | Resume model weights |
| Optimizer state dict | Resume momentum, adaptive rates |
| Scheduler state dict | Resume learning rate schedule |
| Epoch / global step | Track progress |
| RNG states | Reproducibility |
| Best metrics | Track best model |
| Data loader state | Resume data iteration (optional) |

### When to Checkpoint

- **Every N epochs** (e.g., every 1-5 epochs)
- **Every N steps** (e.g., every 1000 steps)
- **On signal** (SIGTERM, SIGUSR1 for preemption)
- **On best validation** (save best model separately)

---

## 📁 Templates

| File | Description |
|------|-------------|
| [checkpoint_utils.py](templates/checkpoint_utils.py) | Reusable checkpointing class |
| [train_with_checkpoint.py](templates/train_with_checkpoint.py) | Example training script |
| [auto_resume.sbatch](templates/auto_resume.sbatch) | Self-resubmitting job script |

---

## 🔧 Quick Reference

### Basic Checkpointing

```python
import torch

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']
```

### Signal Handling

```python
import signal
import sys

def signal_handler(signum, frame):
    print(f"Received signal {signum}, saving checkpoint...")
    save_checkpoint(model, optimizer, scheduler, epoch, 'checkpoint_preempt.pt')
    sys.exit(0)

# Register handlers
signal.signal(signal.SIGTERM, signal_handler)  # Slurm preemption
signal.signal(signal.SIGUSR1, signal_handler)  # Custom signal
```

### DDP Checkpointing

```python
import torch.distributed as dist

def save_checkpoint_ddp(model, optimizer, epoch, rank, path):
    # Only save on rank 0
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # Note: .module for DDP
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
    
    # Ensure all ranks wait for save
    dist.barrier()
```

---

## 📖 Further Reading

- [PyTorch Checkpointing Tutorial](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [FSDP Checkpointing](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.state_dict)
