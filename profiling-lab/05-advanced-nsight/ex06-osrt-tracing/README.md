# Exercise 06: OS Runtime Tracing

## Learning Objectives
- Enable OS runtime tracing for system-level insights
- Profile file I/O, memory allocation, and syscalls
- Identify system-level bottlenecks
- Understand thread scheduling and context switches

## Background

OS runtime tracing captures low-level system events that affect GPU performance:
- File operations (data loading)
- Memory allocation/deallocation
- Thread creation and synchronization
- Network operations

---

## Part 1: Enable OS Runtime Tracing

```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --osrt-events=all \
    -o osrt_profile \
    python train.py
```

### Event Types:
| Event Type | What It Captures |
|------------|------------------|
| `file_io` | File open, read, write, close |
| `socket_io` | Network operations |
| `memory` | malloc, free, mmap |
| `thread` | Thread creation, joins |
| `sync` | Mutex, semaphore operations |
| `all` | Everything |

---

## Part 2: Focused Tracing

```bash
# File I/O only (for data loading analysis)
nsys profile \
    --trace=cuda,osrt \
    --osrt-events=file_io \
    -o file_io_profile \
    python train.py

# Memory operations
nsys profile \
    --trace=cuda,osrt \
    --osrt-events=memory \
    -o memory_profile \
    python train.py
```

---

## Part 3: Analyzing OS Events

### In Timeline:
1. Look for "OS Runtime" rows
2. File I/O shows as read/write blocks
3. Correlate with GPU idle time

### Using CLI:
```bash
# OS runtime summary
nsys stats --report osrt_sum osrt_profile.nsys-rep

# File I/O specific
nsys stats --report file_io osrt_profile.nsys-rep
```

---

## Part 4: Common OS-Level Bottlenecks

### File I/O Bottleneck
```
Pattern:
  OS: [read][read][read][read]....
  GPU:                     [kernel][kernel]
                          ↑
              Waiting for data!
```

**Solution:** Prefetch, more workers, faster storage

### Memory Allocation Overhead
```
Pattern:
  OS: [malloc][malloc][free][malloc][free]...
  GPU:       [kernel]      [kernel]
```

**Solution:** Memory pools, pre-allocation

### Thread Contention
```
Pattern:
  Thread 1: [work][wait][wait][work]
  Thread 2: [wait][work][wait][work]
  Lock: [held][held][held][held]
```

**Solution:** Reduce lock scope, use lock-free structures

---

## Part 5: Hands-On Exercise

### Profile Data Loading Pipeline

```python
"""data_loading_osrt.py - Profile file I/O patterns"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class FileDataset(Dataset):
    def __init__(self, data_dir, num_files=100):
        self.files = []
        # Create test files
        Path(data_dir).mkdir(exist_ok=True)
        for i in range(num_files):
            path = Path(data_dir) / f"data_{i}.pt"
            if not path.exists():
                torch.save(torch.randn(1024), path)
            self.files.append(path)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # This triggers file I/O - profile it!
        return torch.load(self.files[idx])

# Profile this
dataset = FileDataset("/tmp/osrt_test")
loader = DataLoader(dataset, batch_size=16, num_workers=0)

for batch in loader:
    # Simulate GPU work
    if torch.cuda.is_available():
        batch = batch.cuda()
        result = batch.sum()
```

### Profile:
```bash
nsys profile \
    --trace=cuda,osrt \
    --osrt-events=file_io \
    -o data_io_profile \
    python data_loading_osrt.py
```

### Questions to Answer:
1. How much time is spent in file read operations?
2. Does file I/O overlap with GPU computation?
3. What's the read pattern (sequential vs random)?

---

## Part 6: Context Switch Analysis

```bash
# Track context switches
nsys profile \
    --trace=cuda,osrt \
    --cpuctxsw=process-tree \
    -o ctxsw_profile \
    python train.py
```

### High context switches indicate:
- Too many threads competing
- I/O-bound operations
- Lock contention

---

## Reference: OSRT Event Categories

| Category | Events |
|----------|--------|
| **File I/O** | open, close, read, write, lseek, mmap |
| **Memory** | malloc, free, realloc, mmap, munmap |
| **Thread** | pthread_create, pthread_join |
| **Sync** | pthread_mutex_*, pthread_cond_*, sem_* |
| **Network** | socket, connect, send, recv |

---

## Success Criteria

- [ ] Can enable OS runtime tracing
- [ ] Can identify file I/O patterns in timeline
- [ ] Can correlate OS events with GPU activity
- [ ] Understand when OS tracing is useful
