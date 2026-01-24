# Exercise 03: Distributed Training Profiling

## Learning Objectives
- Profile DDP/FSDP training with PyTorch Profiler
- Identify communication overhead
- Analyze compute-communication overlap
- Debug scaling inefficiencies

## Background

Distributed training adds complexity:
- Gradient synchronization (AllReduce)
- Parameter sharding (FSDP)
- Communication overhead
- Load balancing across GPUs

## Part 1: Profile DDP Training

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def profile_ddp_training():
    local_rank = setup_ddp()
    
    model = MyModel().cuda()
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters())
    
    # Profile with different settings per rank
    log_dir = f"./logs/rank_{local_rank}"
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=2, warmup=2, active=6, repeat=1
        ),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        
        for step, batch in enumerate(dataloader):
            if step >= 10:
                break
            
            output = model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            prof.step()
    
    dist.destroy_process_group()
```

### Launch profiling:
```bash
torchrun --nproc_per_node=4 profile_ddp.py
tensorboard --logdir=./logs
```

## Part 2: Analyze Communication

### What to look for in TensorBoard:

1. **NCCL Operations**
   - Look for `ncclAllReduce` events
   - Measure duration vs compute time

2. **Overlap**
   - Check if AllReduce overlaps with backward
   - DDP buckets gradients for overlap

3. **GPU Utilization**
   - Compare across ranks
   - Should be similar (balanced)

### Manual timing:
```python
import time

class TimingCallback:
    def __init__(self):
        self.forward_time = 0
        self.backward_time = 0
        self.sync_time = 0
    
    def step(self, model, loss):
        torch.cuda.synchronize()
        
        # Forward
        t0 = time.perf_counter()
        output = model(input)
        loss = criterion(output, target)
        torch.cuda.synchronize()
        self.forward_time = time.perf_counter() - t0
        
        # Backward (includes AllReduce)
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        self.backward_time = time.perf_counter() - t0
```

## Part 3: Profile FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=local_rank,
)

# Profile FSDP-specific events
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Training loop
    ...

# Look for FSDP events
for event in prof.key_averages():
    if 'fsdp' in event.key.lower() or 'allgather' in event.key.lower():
        print(f"{event.key}: {event.cuda_time_total / 1000:.2f} ms")
```

### FSDP-specific metrics:
- `_all_gather`: Parameter gathering before forward
- `_reduce_scatter`: Gradient reduction after backward
- Memory usage per GPU (should be reduced)

## Part 4: Communication Overhead Analysis

```python
def measure_communication_overhead(model, dataloader, num_steps=10):
    """Measure ratio of communication to computation."""
    
    # Disable DDP gradient sync
    model.require_backward_grad_sync = False
    
    compute_times = []
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        
        torch.cuda.synchronize()
        compute_times.append(time.perf_counter() - start)
    
    avg_compute = sum(compute_times) / len(compute_times)
    
    # Enable sync and measure full time
    model.require_backward_grad_sync = True
    
    full_times = []
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        
        torch.cuda.synchronize()
        full_times.append(time.perf_counter() - start)
    
    avg_full = sum(full_times) / len(full_times)
    
    comm_overhead = avg_full - avg_compute
    comm_ratio = comm_overhead / avg_full * 100
    
    print(f"Compute time: {avg_compute*1000:.2f} ms")
    print(f"Communication: {comm_overhead*1000:.2f} ms")
    print(f"Comm overhead: {comm_ratio:.1f}%")
    
    return avg_compute, comm_overhead
```

## Part 5: Scaling Efficiency

```python
def calculate_scaling_efficiency(single_gpu_time, multi_gpu_time, num_gpus):
    """
    Calculate strong scaling efficiency.
    
    Ideal: multi_gpu_time = single_gpu_time / num_gpus
    Efficiency = ideal_time / actual_time
    """
    ideal_time = single_gpu_time / num_gpus
    efficiency = ideal_time / multi_gpu_time * 100
    speedup = single_gpu_time / multi_gpu_time
    
    print(f"Single GPU: {single_gpu_time*1000:.2f} ms")
    print(f"{num_gpus} GPUs: {multi_gpu_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x (ideal: {num_gpus}x)")
    print(f"Efficiency: {efficiency:.1f}%")
    
    return efficiency
```

## Exercise Tasks

1. **Profile 2-GPU DDP training**
   - Capture traces for both ranks
   - Compare in TensorBoard

2. **Measure communication overhead**
   - Calculate compute vs communication ratio
   - Identify if communication is the bottleneck

3. **Test scaling efficiency**
   - Profile with 1, 2, 4 GPUs
   - Plot scaling curve

4. **Optimize overlap**
   - Tune bucket size
   - Check if backward overlaps with AllReduce

## Expected Results

| Metric | 1 GPU | 2 GPUs | 4 GPUs |
|--------|-------|--------|--------|
| Step time | 100 ms | 55 ms | 30 ms |
| Speedup | 1x | 1.8x | 3.3x |
| Efficiency | 100% | 90% | 83% |
| Comm overhead | 0% | 10% | 17% |

## Common Issues

1. **Poor scaling**: Check data loading, increase batch size
2. **Unbalanced GPUs**: Profile each rank separately
3. **No overlap**: Model too small, increase bucket size
4. **NCCL errors**: Check NCCL_DEBUG=INFO output

## Success Criteria

- [ ] Profiled multi-GPU training with traces per rank
- [ ] Measured communication overhead percentage
- [ ] Calculated scaling efficiency
- [ ] Identified optimization opportunities
