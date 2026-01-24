# Exercise 04: Multi-GPU & NCCL Timeline Analysis

## Learning Objectives
- Profile multi-GPU workloads with Nsight Systems
- Understand NCCL collective communication patterns
- Identify communication bottlenecks in distributed training
- Optimize compute-communication overlap

## Background

Multi-GPU training relies on collective communication (AllReduce, AllGather, etc.) to synchronize gradients. NCCL (NVIDIA Collective Communications Library) provides optimized primitives.

### Common NCCL Operations
| Operation | Purpose | Pattern |
|-----------|---------|---------|
| AllReduce | Sum gradients across GPUs | Ring/Tree |
| AllGather | Collect tensors from all GPUs | Ring |
| ReduceScatter | Reduce + distribute chunks | Ring |
| Broadcast | Send from one to all | Tree |

## Prerequisites

- Multi-GPU system (2+ GPUs)
- NCCL installed
- PyTorch with distributed support

## Exercise Files

```
ex04-multi-gpu-timeline/
├── ddp_training.py     # DDP training script
├── profile_ddp.sh      # Profiling wrapper
├── analyze_nccl.py     # Parse NCCL events
└── README.md
```

## Part 1: Profile DDP Training

```bash
# Profile 2-GPU training
nsys profile --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    -o ddp_profile \
    torchrun --nproc_per_node=2 ddp_training.py

# Open timeline
nsys-ui ddp_profile.nsys-rep
```

### What to observe in timeline:
1. **GPU rows**: Each GPU has its own row
2. **NCCL kernels**: Look for `ncclKernel` operations
3. **Compute gaps**: Time waiting for communication
4. **Overlap**: Forward pass overlapping with gradient sync

## Part 2: Understanding the Timeline

```
GPU 0: [Forward] [Backward] [===AllReduce===] [Optimizer]
GPU 1: [Forward] [Backward] [===AllReduce===] [Optimizer]
                            ↑
                    Communication sync point
```

### Key Questions:
1. How long is AllReduce vs computation?
2. Is gradient sync overlapping with backward pass?
3. Are GPUs balanced (similar kernel times)?

## Part 3: Enable NCCL Debug Logging

```bash
# Set environment variables before profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Profile with NCCL logging
nsys profile ... torchrun --nproc_per_node=2 ddp_training.py 2>&1 | tee nccl.log
```

### NCCL debug output shows:
- Algorithm selection (Ring, Tree, etc.)
- Channel count
- Actual data transfer sizes

## Part 4: Analyze Communication Overhead

```python
# Calculate communication efficiency
compute_time = 100  # ms (from profiler)
comm_time = 30      # ms (AllReduce duration)
total_time = 115    # ms (includes overlap)

# Overlap efficiency
overlap = (compute_time + comm_time - total_time) / comm_time * 100
print(f"Compute-communication overlap: {overlap:.0f}%")

# Scaling efficiency
single_gpu_time = 100  # ms
two_gpu_time = 60      # ms
ideal_speedup = 2.0
actual_speedup = single_gpu_time / two_gpu_time
efficiency = actual_speedup / ideal_speedup * 100
print(f"Scaling efficiency: {efficiency:.0f}%")
```

## Part 5: Optimization Techniques

### 1. Gradient Bucketing
```python
# PyTorch DDP automatically buckets gradients
# Tune bucket size for your model
model = DDP(model, bucket_cap_mb=25)  # Default is 25 MB
```

### 2. Overlap Communication
```python
# DDP overlaps AllReduce with backward by default
# Ensure backward pass is long enough
```

### 3. Reduce Communication Volume
```python
# Use gradient compression
# Use mixed precision (FP16 gradients = half the data)
```

## Profiler Tips for Multi-GPU

```bash
# Profile specific rank only
CUDA_VISIBLE_DEVICES=0 nsys profile -o rank0 ...

# Profile all ranks with separate files
nsys profile -o profile_rank%q{RANK} ...

# Limit profile duration (avoid huge files)
nsys profile --duration=30 ...

# Focus on NCCL
nsys profile --trace=cuda,nvtx --cuda-memory-usage=false ...
```

## Success Criteria

- [ ] Can identify NCCL operations in timeline
- [ ] Measured communication overhead percentage
- [ ] Identified overlap (or lack thereof)
- [ ] Know how to improve scaling efficiency

## Common Issues

1. **No overlap**: Check if model is too small
2. **Unbalanced GPUs**: Data loading issues
3. **Slow AllReduce**: Check NVLink vs PCIe topology
4. **Timeout errors**: Increase NCCL_TIMEOUT
