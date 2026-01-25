#!/usr/bin/env python3
"""
fsdp_memory_efficient.py - Memory-efficient FSDP training techniques

Demonstrates:
- CPU offloading for extremely large models
- Gradient accumulation with FSDP
- Selective layer wrapping
- Memory profiling and optimization
- Rate limiters and memory management

Usage:
    torchrun --nproc_per_node=4 fsdp_memory_efficient.py

Author: CUDA Lab
"""

import os
import gc
import time
import argparse
import functools
from typing import Optional, Callable, Set, Type
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
    ModuleWrapPolicy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


@dataclass
class MemoryConfig:
    """Memory optimization configuration."""
    # Sharding
    sharding_strategy: str = "full_shard"
    
    # Precision
    use_mixed_precision: bool = True
    param_dtype: str = "bf16"  # bf16, fp16, fp32
    
    # Offloading
    cpu_offload: bool = False
    
    # Activation checkpointing
    use_activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = 1  # Checkpoint every N layers
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 4
    
    # Memory management
    limit_all_gathers: bool = True
    forward_prefetch: bool = True
    backward_prefetch: str = "backward_pre"  # backward_pre, backward_post
    
    # Rate limiting
    sync_module_states: bool = True


class MemoryTracker:
    """Track GPU memory usage during training."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.peak_memory = 0
        self.snapshots = []
    
    def snapshot(self, label: str):
        """Take a memory snapshot."""
        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        
        self.snapshots.append({
            "label": label,
            "allocated_gb": allocated / 1e9,
            "reserved_gb": reserved / 1e9,
        })
        
        self.peak_memory = max(self.peak_memory, allocated)
    
    def reset_peak(self):
        """Reset peak memory stats."""
        torch.cuda.reset_peak_memory_stats(self.device)
        self.peak_memory = 0
    
    def get_peak_gb(self) -> float:
        """Get peak memory in GB."""
        return torch.cuda.max_memory_allocated(self.device) / 1e9
    
    def print_report(self, rank: int):
        """Print memory usage report."""
        if rank != 0:
            return
        
        print("\n" + "=" * 60)
        print("MEMORY USAGE REPORT")
        print("=" * 60)
        
        for snap in self.snapshots:
            print(f"{snap['label']:30s}: {snap['allocated_gb']:.2f} GB allocated, "
                  f"{snap['reserved_gb']:.2f} GB reserved")
        
        print("-" * 60)
        print(f"Peak Memory: {self.get_peak_gb():.2f} GB")
        print("=" * 60)


# Model components
class TransformerBlock(nn.Module):
    """Standard transformer block."""
    
    def __init__(self, d_model: int = 1024, nhead: int = 16, dim_ff: int = 4096):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(0.1),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Pre-norm architecture
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class LargeTransformer(nn.Module):
    """Large transformer model for memory testing."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 2048,
        num_layers: int = 24,
        nhead: int = 16,
        dim_ff: int = 8192,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embed.weight
    
    def forward(self, x, labels=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        
        return {"logits": logits, "loss": loss}


def get_mixed_precision_policy(param_dtype: str) -> Optional[MixedPrecision]:
    """Get mixed precision policy."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    
    dtype = dtype_map.get(param_dtype, torch.bfloat16)
    
    if dtype == torch.float32:
        return None
    
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )


def get_cpu_offload_policy(enable: bool) -> Optional[CPUOffload]:
    """Get CPU offload policy."""
    if not enable:
        return None
    
    return CPUOffload(offload_params=True)


def apply_selective_activation_checkpointing(
    model: FSDP,
    checkpoint_every_n: int = 1,
) -> FSDP:
    """Apply activation checkpointing to every N-th layer."""
    layer_count = [0]
    
    def check_fn(submodule):
        if isinstance(submodule, TransformerBlock):
            layer_count[0] += 1
            return layer_count[0] % checkpoint_every_n == 0
        return False
    
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=check_fn,
    )
    
    return model


def wrap_model_with_fsdp(
    model: nn.Module,
    config: MemoryConfig,
    rank: int,
) -> FSDP:
    """Wrap model with FSDP using memory-efficient settings."""
    
    # Sharding strategy
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    
    # Backward prefetch
    prefetch_map = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    
    # Auto wrap policy
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    
    fsdp_kwargs = {
        "sharding_strategy": strategy_map[config.sharding_strategy],
        "mixed_precision": get_mixed_precision_policy(config.param_dtype) if config.use_mixed_precision else None,
        "cpu_offload": get_cpu_offload_policy(config.cpu_offload),
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": prefetch_map[config.backward_prefetch],
        "forward_prefetch": config.forward_prefetch,
        "limit_all_gathers": config.limit_all_gathers,
        "sync_module_states": config.sync_module_states,
        "use_orig_params": True,
        "device_id": torch.cuda.current_device(),
    }
    
    model = FSDP(model, **fsdp_kwargs)
    
    # Apply activation checkpointing
    if config.use_activation_checkpointing:
        model = apply_selective_activation_checkpointing(
            model,
            config.checkpoint_every_n_layers,
        )
        if rank == 0:
            print(f"Applied activation checkpointing (every {config.checkpoint_every_n_layers} layer(s))")
    
    return model


def train_step_with_gradient_accumulation(
    model: nn.Module,
    batch: tuple,
    optimizer: optim.Optimizer,
    config: MemoryConfig,
    accumulation_step: int,
) -> float:
    """Training step with gradient accumulation."""
    input_ids, labels = batch
    input_ids = input_ids.cuda()
    labels = labels.cuda()
    
    # Scale loss by accumulation steps
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"] / config.gradient_accumulation_steps
    
    # Backward
    loss.backward()
    
    # Step optimizer after accumulation
    if (accumulation_step + 1) % config.gradient_accumulation_steps == 0:
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item() * config.gradient_accumulation_steps


def estimate_memory_requirement(
    num_params: int,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_layers: int,
    dtype_bytes: int = 2,  # bf16
    optimizer_states: int = 2,  # AdamW has 2 states per param
) -> dict:
    """Estimate memory requirements."""
    
    # Model memory (parameters)
    param_memory = num_params * dtype_bytes
    
    # Optimizer states
    optim_memory = num_params * 4 * optimizer_states  # Always fp32
    
    # Gradient memory
    grad_memory = num_params * dtype_bytes
    
    # Activation memory (rough estimate)
    # Each layer stores: attention scores + intermediate activations
    activation_per_layer = batch_size * seq_len * d_model * dtype_bytes * 4
    activation_memory = activation_per_layer * num_layers
    
    # Total
    total = param_memory + optim_memory + grad_memory + activation_memory
    
    return {
        "params_gb": param_memory / 1e9,
        "optimizer_gb": optim_memory / 1e9,
        "gradients_gb": grad_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "total_gb": total / 1e9,
    }


def compare_memory_strategies(rank: int, world_size: int):
    """Compare different memory optimization strategies."""
    if rank == 0:
        print("\n" + "=" * 60)
        print("COMPARING MEMORY STRATEGIES")
        print("=" * 60)
    
    # Test configurations
    configs = [
        ("Baseline (no optimizations)", MemoryConfig(
            use_mixed_precision=False,
            use_activation_checkpointing=False,
            gradient_accumulation_steps=1,
        )),
        ("Mixed Precision Only", MemoryConfig(
            use_mixed_precision=True,
            use_activation_checkpointing=False,
            gradient_accumulation_steps=1,
        )),
        ("+ Activation Checkpointing", MemoryConfig(
            use_mixed_precision=True,
            use_activation_checkpointing=True,
            gradient_accumulation_steps=1,
        )),
        ("+ Gradient Accumulation (4x)", MemoryConfig(
            use_mixed_precision=True,
            use_activation_checkpointing=True,
            gradient_accumulation_steps=4,
        )),
    ]
    
    results = []
    
    for name, config in configs:
        if rank == 0:
            print(f"\nTesting: {name}")
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Create model
            model = LargeTransformer(
                d_model=1024,
                num_layers=12,
                dim_ff=4096,
            )
            
            # Wrap with FSDP
            model = wrap_model_with_fsdp(model, config, rank)
            
            # Create optimizer
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            
            # Create small batch
            effective_batch = 4 // config.gradient_accumulation_steps
            input_ids = torch.randint(0, 32000, (max(1, effective_batch), 256)).cuda()
            labels = input_ids.clone()
            
            # Training step
            model.train()
            for step in range(config.gradient_accumulation_steps):
                train_step_with_gradient_accumulation(
                    model, (input_ids, labels), optimizer, config, step
                )
            
            # Record memory
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            results.append((name, peak_memory, "✓"))
            
            if rank == 0:
                print(f"  Peak Memory: {peak_memory:.2f} GB")
            
            # Cleanup
            del model, optimizer
            
        except torch.cuda.OutOfMemoryError:
            results.append((name, float('inf'), "OOM"))
            if rank == 0:
                print("  Result: Out of Memory!")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Print comparison
    if rank == 0:
        print("\n" + "=" * 60)
        print("MEMORY COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Strategy':<40} {'Peak Memory':>12} {'Status':>8}")
        print("-" * 60)
        
        baseline = results[0][1] if results[0][1] != float('inf') else 1
        for name, memory, status in results:
            if memory != float('inf'):
                reduction = (1 - memory / baseline) * 100 if baseline > 0 else 0
                print(f"{name:<40} {memory:>9.2f} GB {status:>8} ({reduction:+.0f}%)")
            else:
                print(f"{name:<40} {'N/A':>12} {status:>8}")
        
        print("=" * 60)


def demo_cpu_offload(rank: int, world_size: int):
    """Demonstrate CPU offloading for very large models."""
    if rank == 0:
        print("\n" + "=" * 60)
        print("CPU OFFLOAD DEMONSTRATION")
        print("=" * 60)
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    config = MemoryConfig(
        cpu_offload=True,
        use_mixed_precision=True,
        use_activation_checkpointing=True,
    )
    
    # Create larger model
    model = LargeTransformer(
        d_model=2048,
        num_layers=24,
        dim_ff=8192,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model parameters: {num_params / 1e9:.2f}B")
    
    # Wrap with FSDP + CPU offload
    model = wrap_model_with_fsdp(model, config, rank)
    
    if rank == 0:
        print(f"GPU Memory after FSDP wrap: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Small batch due to model size
    input_ids = torch.randint(0, 32000, (1, 256)).cuda()
    labels = input_ids.clone()
    
    # Forward + backward
    model.train()
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    
    if rank == 0:
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU Memory: {peak:.2f} GB")
        print("\nCPU offload allows training models larger than GPU memory!")
    
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Memory-Efficient FSDP Training")
    parser.add_argument("--mode", type=str, default="compare",
                        choices=["compare", "offload", "all"])
    args = parser.parse_args()
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print("=" * 60)
        print("MEMORY-EFFICIENT FSDP TRAINING")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        if args.mode == "compare" or args.mode == "all":
            compare_memory_strategies(rank, world_size)
        
        if args.mode == "offload" or args.mode == "all":
            demo_cpu_offload(rank, world_size)
    
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
