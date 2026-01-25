# Fault Tolerance

Fault tolerance techniques for reliable distributed training.

## Overview

Training large models on distributed systems requires handling:
- **Node failures** - Hardware or software crashes
- **Network issues** - Communication timeouts
- **Preemption** - Cloud spot instances being terminated
- **Resource changes** - Nodes joining/leaving dynamically

## Components

### 1. Checkpoint Manager (`checkpoint_manager.py`)

Robust checkpointing with:
- Automatic checkpoint saving at intervals
- Keep last N checkpoints (cleanup old ones)
- Track and save best model
- Async saving to not block training
- FSDP checkpoint support

```python
from checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    keep_last_n=3,
    save_best=True,
    best_metric="loss",
)

# Save during training
manager.save(model, optimizer, step=1000, metrics={"loss": 0.5})

# Resume from latest
state = manager.load_latest(model, optimizer)
```

### 2. Elastic Training (`elastic_training.py`)

Elastic training with PyTorch torchrun:
- Dynamic node addition/removal
- Automatic recovery from failures
- State preservation across restarts

```bash
torchrun \
    --nnodes=1:4 \
    --nproc_per_node=4 \
    --rdzv_id=job_123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master:29400 \
    elastic_training.py
```

## Best Practices

### 1. Checkpoint Frequently
```python
# Save every N steps
if step % 1000 == 0:
    manager.save(model, optimizer, step=step)

# Also save on signals (SIGTERM, SIGINT)
signal.signal(signal.SIGTERM, lambda s, f: manager.save(...))
```

### 2. Use Atomic Saves
```python
# Don't do this - can corrupt on failure
torch.save(state, "checkpoint.pt")

# Do this - atomic rename
torch.save(state, "checkpoint.pt.tmp")
os.rename("checkpoint.pt.tmp", "checkpoint.pt")
```

### 3. Handle NCCL Timeouts
```python
import datetime
dist.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(minutes=30),
)
```

### 4. Test Recovery
```bash
# Kill a node during training
kill -9 <pid>

# Training should recover automatically
```

## Integration with Cloud

### AWS Spot Instances
```python
# Handle spot termination notice
import requests

def check_spot_termination():
    try:
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/spot/instance-action",
            timeout=1
        )
        if response.status_code == 200:
            return True
    except:
        pass
    return False

# Check periodically and save checkpoint
if check_spot_termination():
    manager.save(model, optimizer, step=step)
```

### Slurm
```bash
#!/bin/bash
#SBATCH --signal=SIGTERM@120  # Send SIGTERM 2 minutes before timeout
```

```python
# Handle Slurm signal
signal.signal(signal.SIGTERM, save_and_exit)
```
