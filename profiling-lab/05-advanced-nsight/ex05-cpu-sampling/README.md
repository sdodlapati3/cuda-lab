# Exercise 05: CPU Sampling & Bottleneck Detection

## Learning Objectives
- Enable CPU sampling in Nsight Systems
- Identify CPU-side bottlenecks in training loops
- Understand CPU-GPU synchronization points
- Find Python/C++ hotspots

## Background

GPU profiling shows what the GPU is doing, but many performance issues originate on the CPU:
- Data preprocessing
- Model graph construction
- Python overhead
- Synchronization waits

CPU sampling captures stack traces to show where CPU time is spent.

---

## Part 1: Enable CPU Sampling

```bash
# Profile with CPU sampling
nsys profile \
    --trace=cuda,nvtx \
    --sample=cpu \
    --sampling-frequency=1000 \
    --cpuctxsw=process-tree \
    --backtrace=dwarf \
    -o cpu_profile \
    python train.py
```

### Key Flags:
| Flag | Purpose |
|------|---------|
| `--sample=cpu` | Enable CPU sampling |
| `--sampling-frequency=1000` | Samples per second (1ms resolution) |
| `--cpuctxsw=process-tree` | Track context switches |
| `--backtrace=dwarf` | Full stack traces |

---

## Part 2: Understanding CPU Sampling Output

In Nsight Systems GUI:
1. Look at "CPU Sampling" row
2. Each sample shows a stack trace
3. Aggregate view shows hottest functions

### Common CPU Bottlenecks:

**Python Interpreter Overhead**
```
libpython3.x.so:PyEval_EvalFrameDefault  (30%)
  └── training_loop
      └── forward_pass
```

**Data Loading**
```
libc.so:read  (25%)
  └── torch/utils/data/dataloader.py:__next__
      └── load_batch
```

**Tensor Operations on CPU**
```
libtorch_cpu.so:at::native::add  (15%)
  └── preprocessing_fn
```

---

## Part 3: Hands-On Exercise

### Profile This Training Script:

```python
"""training_with_cpu_bottleneck.py"""
import torch
import torch.nn as nn
import time

def inefficient_preprocessing(data):
    """CPU-intensive preprocessing (bottleneck!)"""
    # This runs on CPU and blocks GPU
    result = data.clone()
    for i in range(100):  # Unnecessary iterations
        result = result + 0.001
        result = torch.sin(result)
    return result

def train_step(model, data, target, optimizer, criterion):
    # CPU bottleneck: preprocessing blocks GPU
    data = inefficient_preprocessing(data)  # <-- Find this!
    
    data = data.cuda()
    target = target.cuda()
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### Task:
1. Profile with CPU sampling
2. Find the CPU hotspot
3. Correlate with GPU idle time

```bash
nsys profile \
    --trace=cuda,nvtx \
    --sample=cpu \
    --sampling-frequency=2000 \
    -o bottleneck_profile \
    python training_with_cpu_bottleneck.py
```

---

## Part 4: Analyzing CPU Samples

### Using CLI:
```bash
# Get CPU sampling summary
nsys stats --report sampling_summary cpu_profile.nsys-rep

# Get top functions
nsys stats --report sampling_top_down cpu_profile.nsys-rep
```

### Using Python:
```python
import sqlite3

def analyze_cpu_samples(db_path: str):
    """Analyze CPU sampling data from SQLite export."""
    conn = sqlite3.connect(db_path)
    
    # Get sampling data
    query = """
    SELECT 
        s.value AS function_name,
        COUNT(*) AS sample_count
    FROM SAMPLING_CALL_STACK sc
    JOIN StringIds s ON sc.functionId = s.id
    GROUP BY s.value
    ORDER BY sample_count DESC
    LIMIT 20
    """
    
    import pandas as pd
    df = pd.read_sql_query(query, conn)
    print("Top CPU functions by sample count:")
    print(df)
    
    conn.close()
```

---

## Part 5: Finding CPU-GPU Sync Points

CPU sampling helps identify where code waits for GPU:

### Synchronization Pattern:
```
CPU: [compute][wait.....][compute][wait.....]
GPU:          [kernel]            [kernel]
              ↑ sync               ↑ sync
```

### Look For:
- `cudaDeviceSynchronize` in samples
- `cudaStreamSynchronize`
- `torch.cuda.synchronize` (Python)

```bash
# Profile focusing on synchronization
nsys profile \
    --trace=cuda,sync \
    --sample=cpu \
    -o sync_profile \
    python train.py
```

---

## Part 6: CPU Sampling Best Practices

### 1. Match Sampling to Workload
```bash
# Short runs: higher frequency
--sampling-frequency=5000

# Long runs: lower frequency
--sampling-frequency=500
```

### 2. Enable DWARF for Full Stacks
```bash
# Need debug symbols for best results
--backtrace=dwarf

# Compile with debug info
gcc -g -O2 mycode.c
```

### 3. Profile Representative Workload
```python
# Warmup to avoid profiling JIT compilation
for _ in range(10):
    model(dummy_input)

# Then profile real work
torch.cuda.cudart().cudaProfilerStart()
for batch in dataloader:
    train_step(...)
torch.cuda.cudart().cudaProfilerStop()
```

---

## Part 7: Common CPU Bottlenecks to Find

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| Python loops | High PyEval time | Vectorize with NumPy/Torch |
| Data loading | High I/O wait | More workers, prefetch |
| Preprocessing | CPU compute spikes | Move to GPU or cache |
| Synchronization | cudaSync calls | Use async operations |
| Memory allocation | malloc/new calls | Pre-allocate buffers |
| String operations | String formatting | Reduce logging |

---

## Exercise Files

```
ex05-cpu-sampling/
├── README.md
├── training_with_cpu_bottleneck.py
├── profile_cpu.sh
├── analyze_cpu_samples.py
└── solutions/
    └── optimized_training.py
```

---

## Success Criteria

- [ ] Can enable CPU sampling
- [ ] Can identify CPU hotspots in stack traces
- [ ] Can correlate CPU activity with GPU idle time
- [ ] Found the bottleneck in the example script
