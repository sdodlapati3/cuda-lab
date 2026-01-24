# 03 - PyTorch Profiler

Master PyTorch's built-in profiler for Python-level GPU performance analysis.

## Learning Objectives

After completing this module, you will be able to:
- Use `torch.profiler` to profile training loops
- Identify compute vs data loading bottlenecks
- Understand kernel launch overhead
- Integrate with TensorBoard for visualization
- Compare CPU vs GPU time for operations

## Why PyTorch Profiler?

| Tool | Level | Best For |
|------|-------|----------|
| Nsight Systems | System | Timeline, multi-GPU, CPU-GPU interaction |
| Nsight Compute | Kernel | Detailed kernel metrics, roofline |
| **PyTorch Profiler** | **Python** | **Training loops, data loading, memory** |

PyTorch Profiler excels at:
- **High-level bottleneck identification** - quickly find slow operations
- **Memory profiling** - track tensor allocations
- **Data loader analysis** - identify I/O bottlenecks
- **Training loop optimization** - separate forward/backward/optimizer time

## Prerequisites

```bash
module load python3
crun -p ~/envs/cuda-lab pip install torch tensorboard torch-tb-profiler
```

## Exercises

### Exercise 1: Basic Profiling (ex01-basic-profiling/)

Learn the fundamentals of `torch.profiler`:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("model_inference"):
        output = model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Tasks:**
1. Profile a simple model forward pass
2. Identify the most time-consuming operations
3. Compare CPU time vs CUDA time
4. Use `record_function` to label custom regions

### Exercise 2: Training Loop Profiling (ex02-training-loop/)

Profile a complete training iteration:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,    # Skip first iteration
        warmup=1,  # Warmup iteration
        active=3,  # Profile 3 iterations
        repeat=2   # Repeat 2 times
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, (data, target) in enumerate(train_loader):
        # Training step
        optimizer.zero_grad()
        output = model(data.cuda())
        loss = criterion(output, target.cuda())
        loss.backward()
        optimizer.step()
        
        prof.step()  # Signal next step
```

**Tasks:**
1. Profile forward, backward, and optimizer separately
2. Identify data loading bottlenecks
3. Use TensorBoard to visualize the trace
4. Compare AMP (mixed precision) vs full precision

### Exercise 3: Memory Profiling (ex03-memory-profiling/)

Track GPU memory usage:

```python
with profile(
    activities=[ProfilerActivity.CUDA],
    profile_memory=True,
) as prof:
    model(input)

# Memory timeline
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
```

**Tasks:**
1. Profile memory allocation patterns
2. Identify memory peaks
3. Find memory leaks
4. Optimize batch size based on memory profile

### Exercise 4: Trace Export (ex04-trace-export/)

Export and analyze traces:

```python
# Export to Chrome trace format
prof.export_chrome_trace("trace.json")

# Export to TensorBoard
prof.export_stacks("stacks.txt", "self_cuda_time_total")

# View in Chrome: chrome://tracing
```

**Tasks:**
1. Export trace to Chrome format
2. Analyze trace in TensorBoard
3. Compare traces before/after optimization
4. Create a profiling report

## Key APIs

### ProfilerActivity

```python
ProfilerActivity.CPU    # CPU operations
ProfilerActivity.CUDA   # CUDA kernel time
ProfilerActivity.XPU    # Intel GPU (optional)
```

### Schedule

```python
schedule = torch.profiler.schedule(
    wait=1,      # Steps to skip
    warmup=1,    # Warmup steps
    active=3,    # Steps to profile
    repeat=1     # Number of cycles
)
```

### Output Options

```python
# Table output
prof.key_averages().table(
    sort_by="cuda_time_total",  # or "cpu_time_total", "self_cuda_time_total"
    row_limit=20
)

# Group by input shape
prof.key_averages(group_by_input_shape=True)

# Group by stack
prof.key_averages(group_by_stack_n=5)
```

## Common Patterns

### Pattern 1: Find Data Loading Bottleneck

```python
with record_function("data_loading"):
    data, target = next(iter(train_loader))

with record_function("forward"):
    output = model(data.cuda())

with record_function("backward"):
    loss.backward()
```

### Pattern 2: Compare Implementations

```python
def profile_impl(name, fn, *args):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(100):
            fn(*args)
    
    total_time = sum(e.cuda_time for e in prof.key_averages())
    print(f"{name}: {total_time/1000:.2f} ms")
```

### Pattern 3: Memory Tracking Decorator

```python
def memory_profile(fn):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        result = fn(*args, **kwargs)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory: {peak_mem:.2f} GB")
        return result
    return wrapper
```

## TensorBoard Integration

1. Profile with TensorBoard handler:
```python
on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs')
```

2. Launch TensorBoard:
```bash
tensorboard --logdir=./logs
```

3. Navigate to the PyTorch Profiler tab

### TensorBoard Features:
- **Overview**: Summary statistics
- **Operator View**: Time per operator
- **GPU Kernel View**: Individual kernel times
- **Trace View**: Timeline visualization
- **Memory View**: Memory allocations over time

## Best Practices

1. **Always warm up** - First iterations are not representative
2. **Profile representative data** - Use realistic batch sizes
3. **Profile on target hardware** - Results vary by GPU
4. **Use record_function** - Label important code regions
5. **Compare before/after** - Quantify optimization impact

## Next Steps

After completing PyTorch Profiler:
- [Nsight Systems](../01-nsight-systems/) for system-level analysis
- [Nsight Compute](../02-nsight-compute/) for kernel optimization
- [Energy Profiling](../04-energy-profiling/) for power analysis
