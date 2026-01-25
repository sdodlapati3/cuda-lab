# DDP Basics: DistributedDataParallel

Learn PyTorch's DistributedDataParallel (DDP) - the foundation of distributed training.

## Learning Objectives

- Understand DDP's communication patterns
- Launch multi-GPU and multi-node training
- Measure scaling efficiency
- Debug common DDP issues

---

## How DDP Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    DDP Training Step                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. FORWARD PASS (Independent)                                  │
│     Each GPU processes its own mini-batch                       │
│     ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐                   │
│     │GPU 0│    │GPU 1│    │GPU 2│    │GPU 3│                   │
│     │Batch│    │Batch│    │Batch│    │Batch│                   │
│     │ 0   │    │ 1   │    │ 2   │    │ 3   │                   │
│     └──┬──┘    └──┬──┘    └──┬──┘    └──┬──┘                   │
│        │          │          │          │                       │
│  2. BACKWARD PASS + ALL-REDUCE (Synchronized)                   │
│     Gradients computed locally, then averaged across GPUs       │
│        │          │          │          │                       │
│        └──────────┴────┬─────┴──────────┘                       │
│                        │                                         │
│              ┌─────────▼─────────┐                              │
│              │   ALL-REDUCE      │                              │
│              │  (Ring/Tree)      │                              │
│              │  ∇avg = Σ∇i / N   │                              │
│              └─────────┬─────────┘                              │
│                        │                                         │
│  3. OPTIMIZER STEP (Independent)                                │
│     Each GPU updates with same averaged gradients               │
│     → All replicas stay synchronized                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files in This Module

| File | Description | Difficulty |
|------|-------------|------------|
| `ddp_mnist.py` | Simple end-to-end example | ⭐ Beginner |
| `ddp_training.py` | Production-ready template | ⭐⭐ Intermediate |
| `ddp_benchmark.py` | Scaling efficiency test | ⭐⭐ Intermediate |
| `launch_ddp.sh` | Launch scripts for SLURM | ⭐ Beginner |

---

## Quick Start

### Single Machine, Multiple GPUs

```bash
# Recommended: torchrun
torchrun --nproc_per_node=4 ddp_mnist.py

# Alternative: torch.distributed.launch (deprecated)
python -m torch.distributed.launch --nproc_per_node=4 ddp_mnist.py
```

### Multiple Machines

```bash
# On machine 0 (master)
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    ddp_mnist.py

# On machine 1
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    ddp_mnist.py
```

---

## Core Concepts

### 1. Process Group Initialization

```python
import torch.distributed as dist

# Method 1: Environment variables (torchrun sets these)
dist.init_process_group(backend='nccl')

# Method 2: Explicit (for debugging)
dist.init_process_group(
    backend='nccl',
    init_method='tcp://localhost:29500',
    world_size=4,
    rank=0
)
```

### 2. Model Wrapping

```python
from torch.nn.parallel import DistributedDataParallel as DDP

# Move model to correct device first
local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device(f'cuda:{local_rank}')
model = model.to(device)

# Wrap with DDP
model = DDP(model, device_ids=[local_rank])
```

### 3. Data Distribution

```python
from torch.utils.data import DataLoader, DistributedSampler

# Create distributed sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

# Use sampler in DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)

# Important: Set epoch for proper shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Different shuffle each epoch
    for batch in dataloader:
        ...
```

### 4. Gradient Synchronization

DDP automatically synchronizes gradients during `backward()`. The synchronization happens:

1. Gradients are bucketed by parameter order
2. All-reduce is called per bucket (overlapped with backward)
3. Result is averaged gradient across all processes

```python
# Gradient sync happens automatically
loss.backward()

# To skip sync (for gradient accumulation):
with model.no_sync():
    loss.backward()  # No all-reduce
```

---

## Common Patterns

### Saving Checkpoints (Only Rank 0)

```python
if dist.get_rank() == 0:
    torch.save({
        'model': model.module.state_dict(),  # Note: .module
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, 'checkpoint.pt')

# Ensure all processes wait
dist.barrier()
```

### Loading Checkpoints

```python
# Load on all ranks
map_location = {'cuda:0': f'cuda:{local_rank}'}
checkpoint = torch.load('checkpoint.pt', map_location=map_location)
model.module.load_state_dict(checkpoint['model'])
```

### Logging (Only Rank 0)

```python
def log_message(msg):
    if dist.get_rank() == 0:
        print(msg)

log_message(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

## Debugging Tips

### 1. Environment Variables

```bash
# Enable NCCL debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Detect hangs
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
```

### 2. Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: NCCL timeout` | Process desync | Check all ranks reach same point |
| `Address already in use` | Port conflict | Change `MASTER_PORT` |
| `Connection refused` | Network issue | Check firewall, use correct interface |
| `Different tensor sizes` | Data mismatch | Ensure same batch size across ranks |

### 3. Verifying Sync

```python
# Check all processes have same weights
for name, param in model.named_parameters():
    if dist.get_rank() == 0:
        dist.broadcast(param.data, src=0)
    else:
        ref = param.data.clone()
        dist.broadcast(ref, src=0)
        assert torch.allclose(param.data, ref), f"Mismatch in {name}"
```

---

## Performance Considerations

### 1. Bucket Size

```python
# Default bucket size is 25MB, tune for your model
model = DDP(model, device_ids=[local_rank], bucket_cap_mb=25)
```

### 2. Find Unused Parameters

```python
# If some parameters aren't used in forward pass
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
```

### 3. Static Graph (PyTorch 2.0+)

```python
# For models with fixed computation graph
model = DDP(model, device_ids=[local_rank], static_graph=True)
```

---

## Exercises

1. **Basic**: Run `ddp_mnist.py` on 2 and 4 GPUs, compare throughput
2. **Intermediate**: Modify `ddp_training.py` to add gradient accumulation
3. **Advanced**: Profile with Nsight Systems to visualize communication overlap

---

## Next Steps

After mastering DDP, move to:
- [FSDP](../02-fsdp/) for large models that don't fit in GPU memory
- [DeepSpeed](../03-deepspeed/) for advanced optimization and offloading
