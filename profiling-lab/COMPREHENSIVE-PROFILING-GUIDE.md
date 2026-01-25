# Comprehensive GPU Profiling Guide

A complete guide to profiling GPU workloads from single GPU to multi-node distributed training, including DeepSpeed, FSDP, and production monitoring.

---

## Table of Contents

1. [Profiling Philosophy](#1-profiling-philosophy)
2. [Tool Overview](#2-tool-overview)
3. [Single GPU Profiling](#3-single-gpu-profiling)
4. [Multi-GPU Profiling (DDP)](#4-multi-gpu-profiling-ddp)
5. [DeepSpeed Profiling](#5-deepspeed-profiling)
6. [FSDP Profiling](#6-fsdp-profiling)
7. [Multi-Node Profiling](#7-multi-node-profiling)
8. [End-to-End Pipeline Profiling](#8-end-to-end-pipeline-profiling)
9. [Memory Profiling](#9-memory-profiling)
10. [Communication Profiling](#10-communication-profiling)
11. [Production Monitoring](#11-production-monitoring)
12. [Troubleshooting Guide](#12-troubleshooting-guide)
13. [Code References](#13-code-references)

---

## 1. Profiling Philosophy

### The Two-Question Framework

Every profiling session should answer:

1. **WHERE** is the bottleneck? → Use **Nsight Systems**
2. **WHAT** operation is slow? → Use **PyTorch Profiler**

### Profiling Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROFILING WORKFLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. BASELINE          2. IDENTIFY           3. DEEP DIVE       │
│  ─────────────        ─────────────         ─────────────      │
│  • Run benchmark      • Nsight Systems      • Nsight Compute   │
│  • Measure throughput • Find bottleneck     • Kernel analysis  │
│  • Note GPU util %    • Timeline gaps       • Roofline model   │
│                                                                 │
│  4. OPTIMIZE          5. VALIDATE           6. MONITOR         │
│  ─────────────        ─────────────         ─────────────      │
│  • Apply fix          • Re-profile          • DCGM metrics     │
│  • One change at time • Compare baseline    • Production logs  │
│  • Document change    • Iterate             • Alerts           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Metrics to Track

| Metric | Target | Tool |
|--------|--------|------|
| GPU Utilization | >80% | nvidia-smi, DCGM |
| GPU Memory | <90% peak | PyTorch Profiler |
| Throughput (samples/sec) | Maximize | Custom timing |
| Time per step | Minimize | PyTorch Profiler |
| Communication overhead | <20% | Nsight Systems |
| Data loading time | <10% of step | Nsight Systems |

---

## 2. Tool Overview

### Decision Matrix

```
┌─────────────────────┬───────────────┬─────────────────┬──────────────────┐
│ Tool                │ Best For      │ Overhead        │ Ease of Use      │
├─────────────────────┼───────────────┼─────────────────┼──────────────────┤
│ Nsight Systems      │ Timeline/gaps │ Low (1-5%)      │ Medium           │
│ PyTorch Profiler    │ Op breakdown  │ Medium (5-15%)  │ Easy             │
│ Nsight Compute      │ Kernel detail │ High (10x+)     │ Hard             │
│ DeepSpeed Profiler  │ FLOPs/memory  │ Low             │ Easy             │
│ DCGM                │ Production    │ Negligible      │ Easy             │
│ torch.cuda.Event    │ Custom timing │ Negligible      │ Easy             │
└─────────────────────┴───────────────┴─────────────────┴──────────────────┘
```

### When to Use Each Tool

| Scenario | Primary Tool | Secondary |
|----------|--------------|-----------|
| "Training is slow" | Nsight Systems | PyTorch Profiler |
| "Out of memory" | PyTorch Profiler | nvidia-smi |
| "Poor scaling" | Nsight Systems (NCCL) | DeepSpeed Profiler |
| "Kernel optimization" | Nsight Compute | Nsight Systems |
| "Production monitoring" | DCGM | Prometheus/Grafana |
| "DeepSpeed tuning" | DeepSpeed Profiler | Nsight Systems |

---

## 3. Single GPU Profiling

### 3.1 Quick Baseline

```python
import torch
import time

def quick_benchmark(model, input_shape, num_iterations=100, warmup=10):
    """Quick throughput benchmark."""
    device = torch.device('cuda')
    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / num_iterations * 1000  # ms
    throughput = input_shape[0] * num_iterations / (end - start)
    
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/sec")
    return avg_time, throughput
```

### 3.2 Nsight Systems

```bash
# Basic profiling
nsys profile -o single_gpu \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    python train.py

# With capture range (for long training)
nsys profile -o single_gpu \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python train.py
```

**In your code:**
```python
import torch

# Start/stop profiling programmatically
torch.cuda.cudart().cudaProfilerStart()
# ... code to profile ...
torch.cuda.cudart().cudaProfilerStop()
```

### 3.3 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=3, active=5, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
        if step >= 10:
            break

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### 3.4 NVTX Annotations

```python
# Add custom markers for Nsight Systems
import torch.cuda.nvtx as nvtx

def train_step(batch):
    nvtx.range_push("data_transfer")
    inputs, labels = batch[0].cuda(), batch[1].cuda()
    nvtx.range_pop()
    
    nvtx.range_push("forward")
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    nvtx.range_pop()
    
    nvtx.range_push("backward")
    loss.backward()
    nvtx.range_pop()
    
    nvtx.range_push("optimizer")
    optimizer.step()
    optimizer.zero_grad()
    nvtx.range_pop()
    
    return loss.item()
```

---

## 4. Multi-GPU Profiling (DDP)

### 4.1 Nsight Systems for DDP

```bash
# Profile all GPUs on single node
nsys profile -o ddp_profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    torchrun --nproc_per_node=4 train_ddp.py
```

### 4.2 PyTorch Profiler for DDP

```python
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

rank = dist.get_rank()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=5),
    on_trace_ready=tensorboard_trace_handler(f'./logs/rank_{rank}'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

### 4.3 Identifying Communication Overhead

```python
import torch
import torch.distributed as dist

class CommTimer:
    """Track DDP communication time."""
    
    def __init__(self):
        self.comm_time = 0.0
        self.compute_time = 0.0
        
    def profile_step(self, model, batch):
        torch.cuda.synchronize()
        
        # Forward + backward (compute)
        start_compute = time.perf_counter()
        loss = model(batch)
        loss.backward()
        torch.cuda.synchronize()
        self.compute_time += time.perf_counter() - start_compute
        
        # All-reduce (communication) - implicit in DDP
        # Measure by comparing with no_sync context
        
        return loss
```

---

## 5. DeepSpeed Profiling

DeepSpeed requires special consideration due to:
- **ZeRO optimizer states** partitioned across GPUs
- **Custom all-reduce/all-gather** communication patterns
- **Activation checkpointing** (gradient checkpointing)
- **Pipeline parallelism** (if used)

### 5.1 DeepSpeed's Built-in Profiler

```python
import deepspeed

# In your ds_config.json
ds_config = {
    "train_batch_size": 32,
    "fp16": {"enabled": True},
    
    # Enable FLOPs profiler
    "flops_profiler": {
        "enabled": True,
        "profile_step": 10,
        "module_depth": -1,
        "top_modules": 3,
        "detailed": True,
        "output_file": "flops_profile.txt"
    }
}

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)

# The profiler automatically captures at profile_step
for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

**Output includes:**
- Total FLOPs
- FLOPs per module
- Parameter count
- Activation memory

### 5.2 DeepSpeed Autotuning

```bash
# Automatically find optimal DeepSpeed configuration
deepspeed --autotuning run train.py \
    --deepspeed ds_config.json
```

**Autotuning config:**
```json
{
    "autotuning": {
        "enabled": true,
        "fast": true,
        "results_dir": "autotuning_results",
        "exps_dir": "autotuning_exps",
        "overwrite": true,
        "metric": "throughput",
        "start_profile_step": 5,
        "end_profile_step": 10
    }
}
```

### 5.3 Nsight Systems with DeepSpeed

```bash
# Profile DeepSpeed training
nsys profile -o deepspeed_profile \
    --trace=cuda,nvtx,osrt \
    deepspeed --num_gpus=4 train.py \
    --deepspeed ds_config.json
```

**Important:** DeepSpeed's communication happens through:
- `torch.distributed` for ZeRO-1/2
- Custom CUDA kernels for ZeRO-3
- NCCL for collective operations

### 5.4 Profiling ZeRO Stages

Each ZeRO stage has different profiling considerations:

```python
"""
ZeRO Stage Profiling Guide:

ZeRO-1 (Optimizer State Partitioning):
- Profile optimizer.step() separately
- Watch for all-gather during step()
- Memory: optimizer states partitioned

ZeRO-2 (+ Gradient Partitioning):
- Profile backward pass communication
- Reduce-scatter after backward
- Memory: gradients partitioned

ZeRO-3 (+ Parameter Partitioning):
- Profile forward AND backward communication
- All-gather before forward, reduce-scatter after backward
- Memory: everything partitioned
- Most communication overhead
"""

def profile_zero_stage(model_engine, batch, stage):
    """Profile a specific ZeRO stage."""
    import torch.cuda.nvtx as nvtx
    
    nvtx.range_push(f"ZeRO-{stage}_forward")
    # ZeRO-3: all-gather happens here
    outputs = model_engine(batch)
    nvtx.range_pop()
    
    nvtx.range_push(f"ZeRO-{stage}_backward")
    # ZeRO-2/3: reduce-scatter happens here
    model_engine.backward(outputs.loss)
    nvtx.range_pop()
    
    nvtx.range_push(f"ZeRO-{stage}_step")
    # ZeRO-1: all-gather for optimizer states
    model_engine.step()
    nvtx.range_pop()
```

### 5.5 DeepSpeed Memory Profiling

```python
import deepspeed
from deepspeed.runtime.utils import memory_status

# Get memory status
def log_memory():
    """Log DeepSpeed memory usage."""
    rank = torch.distributed.get_rank()
    
    # GPU memory
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    
    # DeepSpeed specific
    if hasattr(model_engine, 'optimizer'):
        # ZeRO optimizer memory
        opt_mem = model_engine.optimizer.param_coordinator.total_size / 1e9
    
    print(f"[Rank {rank}] Allocated: {allocated:.2f}GB, "
          f"Reserved: {reserved:.2f}GB")

# Monitor during training
for step, batch in enumerate(dataloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
    
    if step % 10 == 0:
        log_memory()
```

### 5.6 DeepSpeed Communication Analysis

```python
import torch
import torch.distributed as dist

class DeepSpeedCommProfiler:
    """Profile DeepSpeed communication patterns."""
    
    def __init__(self, model_engine):
        self.model_engine = model_engine
        self.comm_times = {
            'all_gather': [],
            'reduce_scatter': [],
            'all_reduce': [],
        }
    
    def profile_step(self, batch):
        """Profile a single training step."""
        torch.cuda.synchronize()
        
        # Hook into DeepSpeed's communication
        # Note: This requires understanding DeepSpeed internals
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        loss = self.model_engine(batch)
        self.model_engine.backward(loss)
        self.model_engine.step()
        end.record()
        
        torch.cuda.synchronize()
        total_time = start.elapsed_time(end)
        
        return total_time
```

### 5.7 Complete DeepSpeed Profiling Script

```python
#!/usr/bin/env python
"""
deepspeed_profiler.py - Comprehensive DeepSpeed profiling

Usage:
    deepspeed --num_gpus=4 deepspeed_profiler.py --model llama-7b
"""

import argparse
import json
import time
import torch
import deepspeed
from pathlib import Path


def create_profiling_config(base_config: dict, profile_step: int = 10) -> dict:
    """Add profiling settings to DeepSpeed config."""
    config = base_config.copy()
    
    # Enable FLOPs profiler
    config["flops_profiler"] = {
        "enabled": True,
        "profile_step": profile_step,
        "module_depth": -1,
        "top_modules": 5,
        "detailed": True,
    }
    
    # Enable wall clock breakdown
    config["wall_clock_breakdown"] = True
    
    return config


def profile_deepspeed_training(
    model,
    train_dataloader,
    ds_config,
    num_steps: int = 50,
    output_dir: str = "./ds_profile"
):
    """Profile DeepSpeed training end-to-end."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    # Timing data
    timings = {
        "forward": [],
        "backward": [],
        "step": [],
        "total": [],
        "throughput": [],
    }
    
    # Training loop with profiling
    model_engine.train()
    
    for step, batch in enumerate(train_dataloader):
        if step >= num_steps:
            break
        
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        
        # Forward
        forward_start = time.perf_counter()
        torch.cuda.nvtx.range_push("forward")
        outputs = model_engine(batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        timings["forward"].append(time.perf_counter() - forward_start)
        
        # Backward
        backward_start = time.perf_counter()
        torch.cuda.nvtx.range_push("backward")
        model_engine.backward(loss)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        timings["backward"].append(time.perf_counter() - backward_start)
        
        # Optimizer step
        step_start_opt = time.perf_counter()
        torch.cuda.nvtx.range_push("optimizer_step")
        model_engine.step()
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        timings["step"].append(time.perf_counter() - step_start_opt)
        
        # Total
        total_time = time.perf_counter() - step_start
        timings["total"].append(total_time)
        
        batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)
        timings["throughput"].append(batch_size * world_size / total_time)
        
        if rank == 0 and step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, "
                  f"time={total_time*1000:.1f}ms, "
                  f"throughput={timings['throughput'][-1]:.1f} samples/s")
    
    # Generate report
    if rank == 0:
        report = generate_report(timings, ds_config, num_steps)
        report_path = output_dir / "deepspeed_profile_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print_report(report)
    
    return timings


def generate_report(timings: dict, ds_config: dict, num_steps: int) -> dict:
    """Generate profiling report."""
    import statistics
    
    def stats(data):
        if len(data) < 2:
            return {"mean": data[0] if data else 0, "std": 0}
        return {
            "mean": statistics.mean(data),
            "std": statistics.stdev(data),
            "min": min(data),
            "max": max(data),
        }
    
    # Skip warmup steps
    warmup = min(5, len(timings["total"]) // 4)
    
    return {
        "config": {
            "zero_stage": ds_config.get("zero_optimization", {}).get("stage", 0),
            "fp16": ds_config.get("fp16", {}).get("enabled", False),
            "bf16": ds_config.get("bf16", {}).get("enabled", False),
            "batch_size": ds_config.get("train_micro_batch_size_per_gpu", 1),
        },
        "timing_ms": {
            "forward": {k: v * 1000 for k, v in stats(timings["forward"][warmup:]).items()},
            "backward": {k: v * 1000 for k, v in stats(timings["backward"][warmup:]).items()},
            "optimizer_step": {k: v * 1000 for k, v in stats(timings["step"][warmup:]).items()},
            "total": {k: v * 1000 for k, v in stats(timings["total"][warmup:]).items()},
        },
        "throughput": stats(timings["throughput"][warmup:]),
        "breakdown_pct": {
            "forward": statistics.mean(timings["forward"][warmup:]) / statistics.mean(timings["total"][warmup:]) * 100,
            "backward": statistics.mean(timings["backward"][warmup:]) / statistics.mean(timings["total"][warmup:]) * 100,
            "optimizer_step": statistics.mean(timings["step"][warmup:]) / statistics.mean(timings["total"][warmup:]) * 100,
        },
        "num_steps": num_steps,
        "warmup_steps": warmup,
    }


def print_report(report: dict):
    """Print formatted profiling report."""
    print("\n" + "="*60)
    print("DeepSpeed Profiling Report")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  ZeRO Stage: {report['config']['zero_stage']}")
    print(f"  FP16: {report['config']['fp16']}")
    print(f"  BF16: {report['config']['bf16']}")
    print(f"  Micro Batch Size: {report['config']['batch_size']}")
    
    print(f"\nTiming (ms):")
    for phase, stats in report['timing_ms'].items():
        print(f"  {phase}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    print(f"\nBreakdown:")
    for phase, pct in report['breakdown_pct'].items():
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {phase:15s} [{bar}] {pct:.1f}%")
    
    print(f"\nThroughput: {report['throughput']['mean']:.1f} ± "
          f"{report['throughput']['std']:.1f} samples/sec")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--config", type=str, default="ds_config.json")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    
    # Load config and run profiling
    # ... (model and dataloader setup)
    pass
```

---

## 6. FSDP Profiling

PyTorch's Fully Sharded Data Parallel (FSDP) has similar considerations to DeepSpeed ZeRO-3.

### 6.1 FSDP-Specific Profiling

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.profiler import profile, ProfilerActivity

# FSDP wrapping
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # Like ZeRO-3
    use_orig_params=True,
)

# Profile with FSDP
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for batch in dataloader:
        # FSDP all-gathers parameters here
        loss = model(batch)
        # FSDP reduce-scatters gradients here
        loss.backward()
        optimizer.step()
        prof.step()
```

### 6.2 FSDP Memory Tracking

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._debug_utils import _get_fsdp_module_states

def profile_fsdp_memory(model):
    """Profile FSDP memory usage per stage."""
    
    # Before forward
    pre_forward_mem = torch.cuda.memory_allocated()
    
    # After forward (parameters gathered)
    output = model(inputs)
    post_forward_mem = torch.cuda.memory_allocated()
    
    # After backward (gradients computed)
    output.loss.backward()
    post_backward_mem = torch.cuda.memory_allocated()
    
    print(f"Memory change during forward: "
          f"{(post_forward_mem - pre_forward_mem) / 1e9:.2f} GB")
    print(f"Memory change during backward: "
          f"{(post_backward_mem - post_forward_mem) / 1e9:.2f} GB")
```

---

## 7. Multi-Node Profiling

### 7.1 SLURM Integration

```bash
#!/bin/bash
#SBATCH --job-name=profile_multinode
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00

# Profile each node separately
srun bash -c '
    export RANK=$SLURM_PROCID
    export LOCAL_RANK=$SLURM_LOCALID
    export WORLD_SIZE=$SLURM_NTASKS
    
    # Create unique profile per node
    nsys profile \
        -o profile_node${SLURM_NODEID}_${SLURM_JOB_ID} \
        --trace=cuda,nvtx,osrt,mpi \
        --stats=true \
        python train_distributed.py \
            --nnodes=$SLURM_NNODES \
            --nproc_per_node=8
'
```

### 7.2 Correlating Multi-Node Traces

```python
"""
Multi-node trace correlation strategy:

1. Use synchronized timestamps (NTP or PTP)
2. Add NVTX markers at epoch/step boundaries
3. Use consistent naming: node_{id}_rank_{rank}
4. Analyze communication patterns across nodes
"""

import torch.distributed as dist

def add_sync_markers(step: int):
    """Add synchronized markers across nodes."""
    # Barrier ensures all nodes at same point
    dist.barrier()
    
    # Add NVTX marker
    torch.cuda.nvtx.range_push(f"step_{step}")
    
    return lambda: torch.cuda.nvtx.range_pop()
```

### 7.3 Inter-Node Communication Analysis

```python
import torch.distributed as dist
import time

def measure_inter_node_bandwidth():
    """Measure inter-node communication bandwidth."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Only measure from rank 0 to rank on another node
    if local_rank != 0:
        return
    
    sizes = [1024, 1024*1024, 100*1024*1024]  # 1KB, 1MB, 100MB
    
    for size in sizes:
        tensor = torch.zeros(size // 4, dtype=torch.float32, device='cuda')
        
        # Warmup
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        
        # Time
        start = time.perf_counter()
        for _ in range(10):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        bandwidth = (size * 2 * 10) / elapsed / 1e9  # GB/s (send + receive)
        
        if rank == 0:
            print(f"Size: {size/1e6:.1f}MB, Bandwidth: {bandwidth:.2f} GB/s")
```

---

## 8. End-to-End Pipeline Profiling

Profile the complete ML pipeline: data loading → preprocessing → training → checkpointing.

### 8.1 Full Pipeline Profiler

```python
import torch
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, ProfilerActivity
from contextlib import contextmanager
import time


class PipelineProfiler:
    """Profile complete training pipeline."""
    
    def __init__(self, output_dir: str = "./pipeline_profile"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timings = {
            "data_loading": [],
            "preprocessing": [],
            "forward": [],
            "backward": [],
            "optimizer": [],
            "checkpoint": [],
            "total_step": [],
        }
        
    @contextmanager
    def phase(self, name: str):
        """Context manager for timing a phase."""
        torch.cuda.synchronize()
        nvtx.range_push(name)
        start = time.perf_counter()
        
        yield
        
        torch.cuda.synchronize()
        nvtx.range_pop()
        elapsed = time.perf_counter() - start
        
        if name in self.timings:
            self.timings[name].append(elapsed)
    
    def profile_epoch(self, model, dataloader, optimizer, criterion):
        """Profile a full training epoch."""
        
        for step, batch in enumerate(dataloader):
            step_start = time.perf_counter()
            
            with self.phase("data_loading"):
                # Data is already loaded by iterator
                pass
            
            with self.phase("preprocessing"):
                inputs, labels = batch
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            with self.phase("forward"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            with self.phase("backward"):
                loss.backward()
            
            with self.phase("optimizer"):
                optimizer.step()
                optimizer.zero_grad()
            
            self.timings["total_step"].append(time.perf_counter() - step_start)
        
        return self.generate_report()
    
    def generate_report(self):
        """Generate pipeline profiling report."""
        report = {}
        total = sum(sum(v) for v in self.timings.values() if v)
        
        for phase, times in self.timings.items():
            if times:
                avg = sum(times) / len(times)
                report[phase] = {
                    "avg_ms": avg * 1000,
                    "percentage": sum(times) / total * 100 if total > 0 else 0,
                }
        
        return report
```

### 8.2 Data Loading Profiling

```python
import torch
from torch.utils.data import DataLoader
import time


class ProfiledDataLoader:
    """DataLoader wrapper with profiling."""
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.load_times = []
        self.transfer_times = []
    
    def __iter__(self):
        for batch in self.dataloader:
            # Time data loading (already done by prefetch)
            load_start = time.perf_counter()
            # Batch is already loaded
            self.load_times.append(time.perf_counter() - load_start)
            
            # Time CPU->GPU transfer
            transfer_start = time.perf_counter()
            if isinstance(batch, (list, tuple)):
                batch = [b.cuda(non_blocking=True) if isinstance(b, torch.Tensor) else b 
                        for b in batch]
            elif isinstance(batch, torch.Tensor):
                batch = batch.cuda(non_blocking=True)
            
            torch.cuda.synchronize()
            self.transfer_times.append(time.perf_counter() - transfer_start)
            
            yield batch
    
    def __len__(self):
        return len(self.dataloader)
    
    def get_stats(self):
        return {
            "avg_load_ms": sum(self.load_times) / len(self.load_times) * 1000,
            "avg_transfer_ms": sum(self.transfer_times) / len(self.transfer_times) * 1000,
        }
```

---

## 9. Memory Profiling

### 9.1 PyTorch Memory Snapshot

```python
import torch

def capture_memory_snapshot(output_path: str = "memory_snapshot.pickle"):
    """Capture detailed memory snapshot."""
    
    # Enable memory history
    torch.cuda.memory._record_memory_history(
        max_entries=100000,
        context_dict=True
    )
    
    # Run your code here
    # ...
    
    # Save snapshot
    torch.cuda.memory._dump_snapshot(output_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    
    print(f"Memory snapshot saved to {output_path}")
    print("View at: https://pytorch.org/memory_viz")
```

### 9.2 Memory Timeline

```python
def memory_timeline(model, dataloader, num_steps=10):
    """Generate memory timeline during training."""
    
    memory_log = []
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        # Log at each phase
        memory_log.append({
            "step": step,
            "phase": "start",
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        })
        
        inputs, labels = batch[0].cuda(), batch[1].cuda()
        memory_log.append({"step": step, "phase": "after_transfer", 
                          "allocated_gb": torch.cuda.memory_allocated() / 1e9})
        
        outputs = model(inputs)
        memory_log.append({"step": step, "phase": "after_forward",
                          "allocated_gb": torch.cuda.memory_allocated() / 1e9})
        
        loss = criterion(outputs, labels)
        loss.backward()
        memory_log.append({"step": step, "phase": "after_backward",
                          "allocated_gb": torch.cuda.memory_allocated() / 1e9})
        
        optimizer.step()
        optimizer.zero_grad()
        memory_log.append({"step": step, "phase": "after_step",
                          "allocated_gb": torch.cuda.memory_allocated() / 1e9})
    
    return memory_log
```

### 9.3 Finding Memory Leaks

```python
import gc
import torch

def find_memory_leaks(model, dataloader, num_steps=100):
    """Detect potential memory leaks."""
    
    initial_mem = torch.cuda.memory_allocated()
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        loss = model(batch[0].cuda()).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Force cleanup
        del loss
        gc.collect()
        torch.cuda.empty_cache()
        
        if step % 10 == 0:
            current_mem = torch.cuda.memory_allocated()
            leak = (current_mem - initial_mem) / 1e6
            print(f"Step {step}: Memory growth = {leak:.2f} MB")
            
            if leak > 100:  # More than 100MB growth
                print("⚠️  Potential memory leak detected!")
```

---

## 10. Communication Profiling

### 10.1 NCCL Communication Profiling

```python
import torch
import torch.distributed as dist
import os

# Enable NCCL debug logging
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

def profile_collective_ops():
    """Profile common collective operations."""
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    results = {}
    
    for size in [1024, 1024*1024, 100*1024*1024]:  # 1KB, 1MB, 100MB
        tensor = torch.ones(size // 4, device='cuda') * rank
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        
        # All-reduce
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            dist.all_reduce(tensor)
        end.record()
        torch.cuda.synchronize()
        
        results[f"all_reduce_{size//1024}KB"] = start.elapsed_time(end) / 10
        
        # All-gather
        output = [torch.zeros_like(tensor) for _ in range(world_size)]
        start.record()
        for _ in range(10):
            dist.all_gather(output, tensor)
        end.record()
        torch.cuda.synchronize()
        
        results[f"all_gather_{size//1024}KB"] = start.elapsed_time(end) / 10
    
    return results
```

### 10.2 Communication/Computation Overlap

```python
class OverlapProfiler:
    """Profile computation/communication overlap efficiency."""
    
    def __init__(self, model):
        self.model = model
        self.compute_only_time = 0
        self.with_comm_time = 0
    
    def profile(self, batch, num_iterations=10):
        """Measure overlap efficiency."""
        
        # Compute only (no gradient sync)
        if hasattr(self.model, 'no_sync'):
            with self.model.no_sync():
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(num_iterations):
                    loss = self.model(batch)
                    loss.backward()
                torch.cuda.synchronize()
                self.compute_only_time = (time.perf_counter() - start) / num_iterations
        
        # With communication
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            loss = self.model(batch)
            loss.backward()
        torch.cuda.synchronize()
        self.with_comm_time = (time.perf_counter() - start) / num_iterations
        
        # Calculate overlap efficiency
        comm_time = self.with_comm_time - self.compute_only_time
        overlap_pct = (1 - comm_time / self.compute_only_time) * 100
        
        return {
            "compute_only_ms": self.compute_only_time * 1000,
            "with_comm_ms": self.with_comm_time * 1000,
            "comm_overhead_ms": comm_time * 1000,
            "overlap_efficiency_pct": max(0, overlap_pct),
        }
```

---

## 11. Production Monitoring

### 11.1 DCGM Integration

```python
import pynvml

class GPUMonitor:
    """Production GPU monitoring."""
    
    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) 
                       for i in range(self.device_count)]
    
    def get_metrics(self):
        """Get current GPU metrics."""
        metrics = []
        
        for i, handle in enumerate(self.handles):
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            metrics.append({
                "gpu_id": i,
                "gpu_util_pct": util.gpu,
                "memory_util_pct": util.memory,
                "memory_used_gb": memory.used / 1e9,
                "memory_total_gb": memory.total / 1e9,
                "power_w": power / 1000,
                "temperature_c": temp,
            })
        
        return metrics
    
    def start_logging(self, interval_sec=1.0, output_file="gpu_metrics.jsonl"):
        """Start continuous logging."""
        import json
        import time
        import threading
        
        self._running = True
        
        def log_loop():
            with open(output_file, 'w') as f:
                while self._running:
                    metrics = self.get_metrics()
                    timestamp = time.time()
                    for m in metrics:
                        m['timestamp'] = timestamp
                        f.write(json.dumps(m) + '\n')
                    f.flush()
                    time.sleep(interval_sec)
        
        self._thread = threading.Thread(target=log_loop, daemon=True)
        self._thread.start()
    
    def stop_logging(self):
        self._running = False
        self._thread.join()
    
    def __del__(self):
        pynvml.nvmlShutdown()
```

### 11.2 Prometheus Metrics

```python
from prometheus_client import Gauge, start_http_server
import pynvml

# Define metrics
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization', ['gpu_id'])
gpu_memory_used = Gauge('gpu_memory_used_bytes', 'GPU memory used', ['gpu_id'])
gpu_power = Gauge('gpu_power_watts', 'GPU power usage', ['gpu_id'])

def update_metrics():
    """Update Prometheus metrics."""
    pynvml.nvmlInit()
    
    for i in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        
        gpu_utilization.labels(gpu_id=str(i)).set(util.gpu)
        gpu_memory_used.labels(gpu_id=str(i)).set(memory.used)
        gpu_power.labels(gpu_id=str(i)).set(power / 1000)

# Start metrics server
start_http_server(8000)

# Update in training loop
while training:
    train_step()
    update_metrics()
```

---

## 12. Troubleshooting Guide

### Common Issues and Solutions

| Issue | Symptom | Tool | Solution |
|-------|---------|------|----------|
| Low GPU utilization | <50% GPU | Nsight Systems | Find CPU bottlenecks, increase batch size |
| Memory OOM | CUDA OOM error | PyTorch Profiler | Enable gradient checkpointing, reduce batch |
| Slow data loading | GPU idle between steps | Nsight Systems | More workers, pin_memory, prefetch |
| Poor scaling | >2x time for 2x GPUs | Nsight Systems | Communication overlap, check NCCL |
| DeepSpeed slow | High comm overhead | DeepSpeed Profiler | Try different ZeRO stage |

### Diagnostic Commands

```bash
# Check GPU topology
nvidia-smi topo -m

# Check NVLink status
nvidia-smi nvlink -s

# Check NCCL environment
echo $NCCL_DEBUG

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check if GPUs can communicate
python -c "import torch.distributed as dist; dist.init_process_group('nccl'); print('OK')"
```

### Performance Checklist

```
□ GPU utilization > 80%?
□ Data loading < 10% of step time?
□ Memory headroom > 10%?
□ Communication overhead < 20%? (distributed)
□ Using mixed precision (FP16/BF16)?
□ torch.compile() enabled? (PyTorch 2.0+)
□ Fused optimizers enabled?
□ Gradient accumulation optimal?
```

---

## 13. Code References

### Profiling Utilities (This Repository)

| File | Description | Use Case |
|------|-------------|----------|
| [unified_profiler.py](utils/unified_profiler.py) | All-in-one profiling script | Single/Multi-GPU/Multi-Node |
| [easy_profiler.py](utils/easy_profiler.py) | Minimal-code profiling decorators | Quick benchmarks |
| [PROFILER-COMPARISON.md](PROFILER-COMPARISON.md) | Tool comparison guide | Choosing the right tool |

### Usage Examples

#### unified_profiler.py

```bash
# Single GPU
python utils/unified_profiler.py train.py

# Multi-GPU
python utils/unified_profiler.py --mode multi-gpu --gpus 4 train.py

# Multi-node
python utils/unified_profiler.py --mode multi-node train.py
```

#### easy_profiler.py

```python
from profiling_lab.utils.easy_profiler import auto_profile, GPUMonitor

# Decorator for automatic profiling
@auto_profile(warmup=3, iterations=10)
def train_step(model, batch):
    return model(batch)

# Context manager for GPU monitoring
with GPUMonitor() as monitor:
    for batch in dataloader:
        train_step(batch)
print(monitor.get_summary())
```

### Nsight Systems Notebooks

| Notebook | Description |
|----------|-------------|
| [01-nsight-systems/](../01-nsight-systems/) | Nsight Systems basics |
| [02-nsight-compute/](../02-nsight-compute/) | Kernel-level profiling |
| [03-pytorch-profiler/](../03-pytorch-profiler/) | PyTorch Profiler guide |
| [04-energy-profiling/](../04-energy-profiling/) | Power consumption analysis |

### External Resources

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [DeepSpeed Profiling](https://www.deepspeed.ai/tutorials/flops-profiler/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/)

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch pynvml tensorboard

# 2. Run quick benchmark
python -c "
from profiling_lab.utils.easy_profiler import quick_benchmark
import torchvision.models as models
model = models.resnet50()
quick_benchmark(model, (32, 3, 224, 224))
"

# 3. Profile with Nsight Systems
nsys profile -o my_profile python train.py

# 4. View results
nsys-ui my_profile.nsys-rep
```

---

*Last updated: January 2026*
