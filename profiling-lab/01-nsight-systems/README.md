# Nsight Systems - Timeline Analysis

> **Learn to read GPU execution timelines and identify performance bottlenecks**

Nsight Systems is NVIDIA's system-wide performance analysis tool. It shows you the **big picture**: when kernels run, when data transfers happen, and where your GPU is sitting idle.

---

## 🎯 Learning Objectives

After completing these exercises, you will be able to:
- Read and interpret Nsight Systems timelines
- Identify GPU idle time and its causes
- Analyze kernel overlap with streams
- Profile distributed training communication

---

## 🔧 Tool Setup

### Installation
Nsight Systems comes with CUDA Toolkit. Verify installation:
```bash
nsys --version
```

### Basic Usage
```bash
# Profile a CUDA application
nsys profile -o report ./my_cuda_app

# Profile a Python script
nsys profile -o report python train.py

# Profile with detailed GPU metrics
nsys profile --trace=cuda,nvtx -o report python train.py
```

### Opening Reports
```bash
# Launch GUI (if available)
nsys-ui report.nsys-rep

# Export to text/json for remote analysis
nsys stats report.nsys-rep
nsys export --type=json report.nsys-rep
```

---

## 📚 Exercises

| Exercise | Topic | Time | Difficulty |
|----------|-------|------|------------|
| [ex01-timeline-basics](ex01-timeline-basics/) | Read timeline, find idle time | 1 hr | ⭐⭐ |
| [ex02-kernel-overlap](ex02-kernel-overlap/) | Streams and concurrency | 1.5 hr | ⭐⭐⭐ |
| [ex03-memory-timeline](ex03-memory-timeline/) | H2D/D2H transfer analysis | 1 hr | ⭐⭐⭐ |
| [ex04-multi-gpu-timeline](ex04-multi-gpu-timeline/) | NCCL communication profiling | 2 hr | ⭐⭐⭐⭐ |

---

## 🔑 Key Concepts

### Timeline Rows
| Row | What It Shows |
|-----|---------------|
| CUDA HW | Actual GPU execution |
| CUDA API | CPU-side CUDA calls |
| Memory | H2D, D2H, D2D transfers |
| NVTX | Custom annotations |
| Streams | Per-stream activity |

### Common Patterns to Recognize

**Good Pattern: Overlapped Execution**
```
Stream 0: [Kernel A][Kernel C][Kernel E]
Stream 1:    [Kernel B][Kernel D]
Memory:   [H2D]          [D2H]
```

**Bad Pattern: Sequential with Gaps**
```
Stream 0: [Kernel A]    [Kernel B]    [Kernel C]
                   ^^^^         ^^^^
                   IDLE         IDLE
```

### Key Metrics to Check
- **GPU Idle %**: Should be <10% for compute-bound workloads
- **Kernel Duration**: Look for outliers
- **Memory Transfer Time**: Should overlap with compute
- **API Call Overhead**: Minimize synchronous calls

---

## 📊 Quick Reference

### Useful nsys Commands
```bash
# Profile with NVTX annotations
nsys profile --trace=cuda,nvtx,osrt -o report ./app

# Limit capture time
nsys profile --duration=60 -o report ./app

# Capture specific process
nsys profile --trace-fork-before-exec=true -o report python train.py

# Generate summary stats
nsys stats report.nsys-rep

# Export timeline data
nsys export --type=sqlite report.nsys-rep
```

### Adding NVTX Markers (Python)
```python
import torch

# Automatic PyTorch profiling
with torch.cuda.nvtx.range("forward_pass"):
    output = model(input)

with torch.cuda.nvtx.range("backward_pass"):
    loss.backward()
```

### Adding NVTX Markers (CUDA C++)
```cpp
#include <nvtx3/nvToolsExt.h>

nvtxRangePush("my_kernel");
my_kernel<<<grid, block>>>(args);
nvtxRangePop();
```

---

## 📖 Further Reading

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NVTX Documentation](https://nvidia.github.io/NVTX/)
- [PyTorch Profiler with Nsight](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
