# FSDP: Fully Sharded Data Parallel

Train models that don't fit in GPU memory using PyTorch's Fully Sharded Data Parallel.

## Learning Objectives

- Understand FSDP sharding strategies
- Configure FSDP for different model sizes
- Implement efficient checkpointing
- Optimize memory usage

---

## How FSDP Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FSDP Architecture                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Traditional DDP (Model replicated):                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │Full Model   │ │Full Model   │ │Full Model   │ │Full Model   │  │
│  │+ Gradients  │ │+ Gradients  │ │+ Gradients  │ │+ Gradients  │  │
│  │+ Optimizer  │ │+ Optimizer  │ │+ Optimizer  │ │+ Optimizer  │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │
│       GPU 0          GPU 1          GPU 2          GPU 3           │
│  Memory: 4x Model Size per GPU                                      │
│                                                                     │
│  FSDP (Parameters sharded):                                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │
│  │ Shard 0     │ │ Shard 1     │ │ Shard 2     │ │ Shard 3     │  │
│  │ (1/4 model) │ │ (1/4 model) │ │ (1/4 model) │ │ (1/4 model) │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │
│       GPU 0          GPU 1          GPU 2          GPU 3           │
│  Memory: ~1x Model Size per GPU                                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                      FSDP Training Step                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. ALL-GATHER parameters (collect from all GPUs)                   │
│     ┌───┐ ┌───┐ ┌───┐ ┌───┐                                        │
│     │ S0│+│ S1│+│ S2│+│ S3│ → Full Parameters                       │
│     └───┘ └───┘ └───┘ └───┘                                        │
│                                                                     │
│  2. FORWARD pass with full parameters                               │
│     Full Model → Output → Loss                                      │
│                                                                     │
│  3. BACKWARD pass (compute gradients)                               │
│     Loss → Gradients                                                │
│                                                                     │
│  4. REDUCE-SCATTER gradients (each GPU gets its shard)              │
│     Full Gradients → ┌───┐ ┌───┐ ┌───┐ ┌───┐                       │
│                      │ G0│ │ G1│ │ G2│ │ G3│                       │
│                      └───┘ └───┘ └───┘ └───┘                       │
│                                                                     │
│  5. OPTIMIZER step (each GPU updates its shard)                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Sharding Strategies

| Strategy | Memory | Communication | Best For |
|----------|--------|---------------|----------|
| `FULL_SHARD` | Lowest | Highest | Large models, memory-constrained |
| `SHARD_GRAD_OP` | Medium | Medium | Medium models, balance |
| `NO_SHARD` | Highest | Lowest | Small models, fast training |
| `HYBRID_SHARD` | Configurable | Configurable | Multi-node, NVLink within node |

### Strategy Details

```python
from torch.distributed.fsdp import ShardingStrategy

# FULL_SHARD: Shard params, grads, and optimizer states
# Memory: O(model_size / world_size)
# Communication: 2x all-gather + reduce-scatter per layer
ShardingStrategy.FULL_SHARD

# SHARD_GRAD_OP: Shard grads and optimizer states only
# Memory: O(model_size + optimizer_states / world_size)  
# Communication: 1x reduce-scatter per layer
ShardingStrategy.SHARD_GRAD_OP

# NO_SHARD: No sharding (like DDP)
# Memory: O(model_size * world_size)
# Communication: 1x all-reduce per layer
ShardingStrategy.NO_SHARD

# HYBRID_SHARD: Shard within node, replicate across nodes
# Best for: NVLink within node + InfiniBand across nodes
ShardingStrategy.HYBRID_SHARD
```

---

## Files in This Module

| File | Description | Difficulty |
|------|-------------|------------|
| `fsdp_basics.py` | Basic FSDP setup and usage | ⭐ Beginner |
| `fsdp_transformer.py` | Training large transformers | ⭐⭐ Intermediate |
| `fsdp_checkpoint.py` | Checkpointing strategies | ⭐⭐ Intermediate |
| `fsdp_memory_efficient.py` | Memory optimization techniques | ⭐⭐⭐ Advanced |

---

## Quick Start

### Basic FSDP

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# Wrap model with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    use_orig_params=True,  # For torch.compile compatibility
)

# Training is the same as DDP!
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### Launch

```bash
torchrun --nproc_per_node=4 fsdp_basics.py
```

---

## Key Configuration Options

### 1. Auto Wrapping Policy

```python
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
import functools

# Wrap layers with >100M parameters
auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy,
    min_num_params=100_000_000,
)

# Or wrap specific transformer layers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer},
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
)
```

### 2. Mixed Precision

```python
from torch.distributed.fsdp import MixedPrecision

# BF16 training (recommended for modern GPUs)
bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# FP16 training
fp16_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

model = FSDP(
    model,
    mixed_precision=bf16_policy,
)
```

### 3. CPU Offloading

```python
from torch.distributed.fsdp import CPUOffload

# Offload parameters to CPU when not in use
model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True),
)
```

---

## Checkpointing

### Full State Dict (For inference or fine-tuning)

```python
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)

# Save full model on rank 0
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, "model.pt")
```

### Sharded State Dict (For resuming training)

```python
from torch.distributed.fsdp import ShardedStateDictConfig

# Each rank saves its shard
sharded_policy = ShardedStateDictConfig(offload_to_cpu=True)

with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_policy):
    state_dict = model.state_dict()
    torch.save(state_dict, f"checkpoint_rank{rank}.pt")
```

---

## Memory Estimation

```python
def estimate_fsdp_memory(
    model_params_billions: float,
    world_size: int,
    mixed_precision: bool = True,
    sharding_strategy: str = "FULL_SHARD"
) -> dict:
    """Estimate GPU memory requirements for FSDP training."""
    
    bytes_per_param = 2 if mixed_precision else 4  # BF16/FP16 vs FP32
    
    # Model parameters
    total_param_bytes = model_params_billions * 1e9 * bytes_per_param
    
    # Optimizer states (Adam: 2x model size in FP32)
    optimizer_bytes = model_params_billions * 1e9 * 4 * 2
    
    # Gradients
    gradient_bytes = total_param_bytes
    
    if sharding_strategy == "FULL_SHARD":
        # Everything sharded
        param_per_gpu = total_param_bytes / world_size
        grad_per_gpu = gradient_bytes / world_size
        opt_per_gpu = optimizer_bytes / world_size
        
        # Temporary all-gathered params during forward/backward
        temp_memory = total_param_bytes  # Full model temporarily
        
    elif sharding_strategy == "SHARD_GRAD_OP":
        # Only grads and optimizer sharded
        param_per_gpu = total_param_bytes
        grad_per_gpu = gradient_bytes / world_size
        opt_per_gpu = optimizer_bytes / world_size
        temp_memory = 0
    
    total_per_gpu = param_per_gpu + grad_per_gpu + opt_per_gpu + temp_memory
    
    return {
        "params_gb": param_per_gpu / 1e9,
        "gradients_gb": grad_per_gpu / 1e9,
        "optimizer_gb": opt_per_gpu / 1e9,
        "temporary_gb": temp_memory / 1e9,
        "total_gb": total_per_gpu / 1e9,
    }
```

---

## Common Issues

### 1. Out of Memory During Forward

**Cause**: All-gather creates full parameter copy temporarily.

**Solution**: Use activation checkpointing:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

# Wrap layers with checkpointing
def apply_activation_checkpointing(model):
    for layer in model.layers:
        checkpoint_wrapper(
            layer,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
```

### 2. Slow Training

**Cause**: Too much communication overhead.

**Solution**: 
- Use `HYBRID_SHARD` for multi-node
- Increase batch size
- Use gradient accumulation

### 3. Checkpoint Size

**Cause**: Full state dict is very large.

**Solution**: Use sharded checkpointing for training, only save full for inference.

---

## Best Practices

1. **Start with FULL_SHARD** - Maximum memory savings
2. **Use BF16 mixed precision** - Better stability than FP16
3. **Enable activation checkpointing** - For very large models
4. **Use transformer_auto_wrap_policy** - Wraps at optimal granularity
5. **Profile memory** - Use `torch.cuda.memory_stats()`

---

## Comparison: FSDP vs DeepSpeed ZeRO

| Feature | FSDP | DeepSpeed ZeRO |
|---------|------|----------------|
| Native PyTorch | ✅ | ❌ (wrapper) |
| Activation checkpointing | ✅ | ✅ |
| CPU offloading | ✅ | ✅ (more mature) |
| NVMe offloading | ❌ | ✅ |
| torch.compile | ✅ | Limited |
| Inference optimization | Limited | ✅ (DeepSpeed-Inference) |

---

## Next Steps

- Try the exercises in `fsdp_basics.py`
- Move to [DeepSpeed](../03-deepspeed/) for additional features
- Learn about [hybrid parallelism](../05-advanced/) for very large models
