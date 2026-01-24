# Exercise 02: Memory Profiling

## Learning Objectives
- Profile GPU memory usage during training
- Identify memory leaks and peak usage
- Optimize memory with gradient checkpointing
- Understand memory timeline

## Background

PyTorch's memory profiler can track:
- Tensor allocations and deallocations
- Peak memory usage
- Memory timeline during forward/backward
- Memory snapshots for debugging

## Part 1: Basic Memory Profiling

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

with profile(
    activities=[ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Memory report
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=20
))
```

## Part 2: Memory Snapshots

```python
# Enable memory history tracking
torch.cuda.memory._record_memory_history(max_entries=100000)

# Run your code
train_step()

# Take snapshot
snapshot = torch.cuda.memory._snapshot()

# Save for visualization
import pickle
with open("memory_snapshot.pkl", "wb") as f:
    pickle.dump(snapshot, f)

# Disable tracking
torch.cuda.memory._record_memory_history(enabled=None)
```

### Analyze with Memory Viz Tool:
```bash
# Generate HTML visualization
python -m torch.cuda.memory._memory_viz trace_plot memory_snapshot.pkl -o memory.html
```

## Part 3: Finding Memory Peaks

```python
import torch

# Track memory manually
def memory_stats():
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
    }

# Reset stats
torch.cuda.reset_peak_memory_stats()

# After each step
for step, batch in enumerate(dataloader):
    # Forward
    output = model(batch)
    print(f"After forward: {memory_stats()}")
    
    # Backward
    loss.backward()
    print(f"After backward: {memory_stats()}")
    
    # Optimizer
    optimizer.step()
    print(f"After optimizer: {memory_stats()}")
    
    optimizer.zero_grad()
    print(f"After zero_grad: {memory_stats()}")
```

## Part 4: Memory Optimization Techniques

### Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
    
    def forward(self, x):
        # Checkpoint expensive blocks
        x = checkpoint(self.block1, x, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        x = self.block3(x)  # Last block not checkpointed
        return x
```

### Profile with/without checkpointing:
```python
# Without checkpointing
torch.cuda.reset_peak_memory_stats()
output = model(input)
loss.backward()
print(f"Without checkpointing: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# With checkpointing
torch.cuda.reset_peak_memory_stats()
output = checkpointed_model(input)
loss.backward()
print(f"With checkpointing: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### Mixed Precision Memory Savings
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

torch.cuda.reset_peak_memory_stats()

with autocast('cuda'):
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print(f"With AMP: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

## Exercise Tasks

1. **Profile your model's memory**
   - Identify peak memory usage
   - Find which operations use most memory

2. **Create memory timeline**
   - Track memory through forward/backward
   - Identify the peak moment

3. **Apply optimizations**
   - Implement gradient checkpointing
   - Enable mixed precision
   - Measure memory reduction

4. **Compare results**
   | Configuration | Peak Memory | Time |
   |--------------|-------------|------|
   | Baseline | ? GB | ? ms |
   | + Checkpointing | ? GB | ? ms |
   | + AMP | ? GB | ? ms |
   | + Both | ? GB | ? ms |

## Analysis Questions

1. Where does memory peak (forward or backward)?
2. How much memory does checkpointing save?
3. What's the compute overhead of checkpointing?
4. Does AMP affect model accuracy?

## Success Criteria

- [ ] Can profile memory usage with PyTorch profiler
- [ ] Generated memory snapshot visualization
- [ ] Reduced peak memory by >30% with optimizations
- [ ] Understand memory/compute tradeoff
