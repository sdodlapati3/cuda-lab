# Distributed Data Loading

This module provides patterns for efficient distributed data loading in multi-GPU and multi-node training.

## Overview

Efficient data loading is critical for maximizing GPU utilization in distributed training.
This module covers:

- **DistributedSampler**: Ensure each GPU sees unique data
- **Multi-process DataLoader**: Overlap CPU preprocessing with GPU compute
- **Sharded Datasets**: Handle datasets too large for single node
- **WebDataset/Mosaic**: Cloud-native dataset formats

## Data Flow in Distributed Training

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED DATA LOADING                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Dataset (D samples)                                                     │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │ sample_0 | sample_1 | sample_2 | ... | sample_D-1             │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│  DistributedSampler (world_size = 4)                                    │
│  ┌───────────────────────────────────────────────────────────────┐      │
│  │ Rank 0: [0, 4, 8, 12, ...]   (every 4th starting at 0)        │      │
│  │ Rank 1: [1, 5, 9, 13, ...]   (every 4th starting at 1)        │      │
│  │ Rank 2: [2, 6, 10, 14, ...]  (every 4th starting at 2)        │      │
│  │ Rank 3: [3, 7, 11, 15, ...]  (every 4th starting at 3)        │      │
│  └───────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│  DataLoader (per GPU, num_workers=4)                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Worker 0    Worker 1    Worker 2    Worker 3                   │    │
│  │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐                 │    │
│  │  │ Load  │   │ Load  │   │ Load  │   │ Load  │  ← Parallel I/O │    │
│  │  │ Augment│  │ Augment│  │ Augment│  │ Augment│ ← CPU preproc  │    │
│  │  └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘                 │    │
│  │      │           │           │           │                      │    │
│  │      └───────────┴───────────┴───────────┘                      │    │
│  │                          │                                       │    │
│  │                          ▼                                       │    │
│  │              Prefetch Queue (pin_memory)                         │    │
│  │              ┌─────────────────────────┐                         │    │
│  │              │ batch_0 | batch_1 | ... │  → Async transfer      │    │
│  │              └─────────────────────────┘                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│                              ▼                                           │
│                          GPU Batch                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Contents

### Scripts
- `distributed_sampler.py` - DistributedSampler patterns and custom samplers
- `efficient_dataloader.py` - Optimized DataLoader configuration
- `webdataset_loader.py` - Cloud-native streaming datasets

## Key Concepts

### 1. DistributedSampler

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # Total number of GPUs
    rank=rank,                 # This GPU's rank
    shuffle=True,              # Shuffle within each epoch
    drop_last=True,            # Drop incomplete batches
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,           # IMPORTANT: No shuffle=True with sampler
    num_workers=4,
    pin_memory=True,
)

# CRITICAL: Update sampler each epoch for different shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # ← Must call this!
    for batch in dataloader:
        ...
```

### 2. DataLoader Optimization

| Parameter | Recommendation | Notes |
|-----------|---------------|-------|
| `num_workers` | 4-8 per GPU | More workers = more RAM |
| `pin_memory` | True | Faster CPU→GPU transfer |
| `prefetch_factor` | 2-4 | Batches to prefetch |
| `persistent_workers` | True | Reuse workers across epochs |
| `drop_last` | True | Avoid uneven batch sizes |

### 3. Memory Pinning

```python
# Automatic with pin_memory=True
dataloader = DataLoader(dataset, pin_memory=True)

# Or manual for custom data
tensor = tensor.pin_memory()  # Enables async transfer
tensor.cuda(non_blocking=True)  # Non-blocking copy
```

## Common Patterns

### Pattern 1: Basic Distributed DataLoader

```python
def create_distributed_dataloader(
    dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    ), sampler
```

### Pattern 2: Sharded Dataset Loading

```python
def load_sharded_dataset(
    shard_paths: List[str],
    rank: int,
    world_size: int,
):
    """Load dataset shards specific to this rank."""
    # Assign shards to ranks
    shards_per_rank = len(shard_paths) // world_size
    my_shards = shard_paths[
        rank * shards_per_rank:(rank + 1) * shards_per_rank
    ]
    
    # Load only this rank's shards
    datasets = [load_shard(path) for path in my_shards]
    return ConcatDataset(datasets)
```

### Pattern 3: Streaming Dataset (WebDataset)

```python
import webdataset as wds

def create_webdataset_loader(
    url: str,
    batch_size: int,
    rank: int,
    world_size: int,
):
    dataset = (
        wds.WebDataset(url)
        .shuffle(1000)
        .decode("pil")
        .to_tuple("jpg", "json")
        .map_tuple(transform, identity)
        .batched(batch_size)
    )
    
    # Shard across workers
    dataset = dataset.with_epoch(10000 // world_size)
    
    return wds.WebLoader(dataset, num_workers=4)
```

## Performance Debugging

### Identify Data Loading Bottleneck

```python
import time

# Measure data loading time
data_times = []
for batch in dataloader:
    start = time.time()
    batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
    torch.cuda.synchronize()
    data_times.append(time.time() - start)

print(f"Avg data time: {sum(data_times)/len(data_times)*1000:.2f}ms")
```

### Check GPU Utilization

```bash
# Should be >90% during training
nvidia-smi dmon -s u -d 1
```

### Profile with PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
) as prof:
    for i, batch in enumerate(dataloader):
        if i > 10:
            break
        # Training step...

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

## Common Issues

### Issue: Stale data after epoch
```python
# WRONG - same data order every epoch
for epoch in range(epochs):
    for batch in dataloader:
        ...

# CORRECT - different shuffle each epoch
for epoch in range(epochs):
    sampler.set_epoch(epoch)  # ← Critical!
    for batch in dataloader:
        ...
```

### Issue: Inconsistent batch sizes
```python
# Use drop_last=True for consistent batch sizes
sampler = DistributedSampler(dataset, drop_last=True)
```

### Issue: OOM in workers
```python
# Reduce workers or prefetch
dataloader = DataLoader(
    dataset,
    num_workers=2,        # Reduce if OOM
    prefetch_factor=1,    # Reduce prefetch
)
```

## Resources

- [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#data-loading)
- [WebDataset Documentation](https://github.com/webdataset/webdataset)
- [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/)
