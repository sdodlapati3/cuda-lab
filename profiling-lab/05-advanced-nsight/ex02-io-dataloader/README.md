# Exercise 02: I/O and DataLoader Profiling

## Learning Objectives
- Profile data loading pipelines with Nsight Systems
- Identify DataLoader bottlenecks (I/O, CPU preprocessing, GPU starving)
- Optimize data pipeline for GPU utilization
- Use file I/O tracing to find disk bottlenecks

## Background

In many ML workloads, the GPU is starved waiting for data. This exercise teaches you to identify and fix data pipeline bottlenecks.

### Common Data Pipeline Issues

```
Ideal:     [Data Load][GPU Train][Data Load][GPU Train]...
                       ↓ Overlap  ↓
Reality:   [Data Load]----[GPU]----[Data Load]----[GPU]...
                      ^^^^        ^^^^
                      GPU IDLE!   GPU IDLE!
```

---

## Exercise Files

```
ex02-io-dataloader/
├── README.md
├── slow_dataloader.py      # Baseline with bottlenecks
├── fast_dataloader.py      # Optimized version
├── profile_dataloader.sh   # Profiling script
├── synthetic_dataset.py    # Dataset for testing
└── analysis.md             # Your analysis notes
```

---

## Part 1: Understanding DataLoader Tracing

Enable file I/O and OS runtime tracing:

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --osrt-events=file_io \
    --sample=cpu \
    -o dataloader_profile \
    python train_with_data.py
```

### Key Traces:
| Trace | What It Shows |
|-------|---------------|
| `osrt` | OS runtime events (file ops, memory) |
| `file_io` | Disk read/write operations |
| `cpu` | CPU sampling for Python overhead |

---

## Part 2: Profile a Slow DataLoader

```python
# slow_dataloader.py - Common mistakes

class SlowDataset(Dataset):
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).glob("*.pt"))
    
    def __getitem__(self, idx):
        # Issue 1: Load from disk every time (no caching)
        data = torch.load(self.files[idx])
        
        # Issue 2: CPU-intensive preprocessing on main thread
        data = self.heavy_preprocess(data)
        
        # Issue 3: Creating new tensors instead of in-place
        data = torch.tensor(data.numpy())  # Unnecessary copy!
        
        return data

# Issue 4: num_workers=0 (no parallel loading)
dataloader = DataLoader(dataset, batch_size=32, num_workers=0)
```

### Profile It:
```bash
./profile_dataloader.sh slow
```

### What to Look For:
1. **Timeline gaps**: Large spaces between GPU kernels
2. **File I/O rows**: Lots of disk reads during training
3. **CPU utilization**: Is CPU preprocessing blocking?

---

## Part 3: Identify the Bottleneck Type

### GPU Starving Pattern
```
Timeline:
CPU: [preprocess.....][preprocess.....][preprocess.....]
GPU:       [train]         [train]         [train]
     ^^^^^         ^^^^^           ^^^^^ 
     IDLE!         IDLE!           IDLE!
```

### I/O Bound Pattern  
```
Timeline:
Disk: [read][read][read][read][read][read]
CPU:       [prep]     [prep]     [prep]
GPU:            [train]    [train]
```

### Well-Optimized Pattern
```
Timeline:
Worker0: [load+prep][load+prep][load+prep]
Worker1: [load+prep][load+prep][load+prep]
GPU:     [train][train][train][train][train]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         Fully utilized!
```

---

## Part 4: Optimize the DataLoader

```python
# fast_dataloader.py - Optimized version

class FastDataset(Dataset):
    def __init__(self, data_dir, cache_in_memory=True):
        self.files = list(Path(data_dir).glob("*.pt"))
        
        # Fix 1: Cache in memory if possible
        if cache_in_memory:
            self.cache = [torch.load(f) for f in self.files]
        else:
            self.cache = None
    
    def __getitem__(self, idx):
        if self.cache:
            data = self.cache[idx]
        else:
            data = torch.load(self.files[idx])
        
        # Fix 2: Lightweight preprocessing only
        return data.float()  # In-place type conversion

# Fix 3: Multiple workers + pinned memory + prefetch
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,           # Parallel loading
    pin_memory=True,         # Faster H2D transfer
    prefetch_factor=2,       # Prefetch batches
    persistent_workers=True  # Keep workers alive
)
```

---

## Part 5: Compare Before/After

```bash
# Profile slow version
./profile_dataloader.sh slow

# Profile fast version  
./profile_dataloader.sh fast

# Compare in GUI
nsys-ui slow_profile.nsys-rep fast_profile.nsys-rep
```

### Metrics to Compare:
| Metric | Slow | Fast | Goal |
|--------|------|------|------|
| GPU Idle % | >30% | <10% | Minimize |
| Samples/sec | Low | High | Maximize |
| Data load time % | >20% | <10% | Minimize |

---

## Part 6: Advanced - Profile DALI Pipeline

For even faster data loading, profile NVIDIA DALI:

```python
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn

@pipeline_def
def dali_pipeline():
    images, labels = fn.readers.file(file_root="data/")
    images = fn.decoders.image(images, device="mixed")  # Decode on GPU!
    images = fn.resize(images, size=[224, 224])
    return images, labels
```

```bash
nsys profile --trace=cuda,nvtx,osrt python train_with_dali.py
```

---

## Hands-On Task

1. Profile `slow_dataloader.py` and identify:
   - What % of time is GPU idle?
   - Where is time spent (disk/CPU/transfer)?

2. Implement fixes in your copy

3. Profile again and document improvement

### Expected Results:
- GPU utilization: 50% → 90%+
- Training throughput: 2-3x improvement

---

## Profiling Commands Reference

```bash
# Full data pipeline profile
nsys profile \
    --trace=cuda,nvtx,osrt \
    --osrt-events=file_io,memory \
    --sample=cpu \
    --python-backtrace=cuda \
    -o data_profile \
    python train.py

# Focus on I/O only
nsys profile \
    --trace=osrt \
    --osrt-events=file_io \
    -o io_profile \
    python train.py

# Get file I/O statistics
nsys stats --report file_io data_profile.nsys-rep
```

---

## Success Criteria

- [ ] Profile shows DataLoader activity in timeline
- [ ] Can identify GPU idle time caused by data loading
- [ ] Measured improvement from optimizations
- [ ] Understand num_workers, pin_memory, prefetch impact
