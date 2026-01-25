#!/usr/bin/env python3
"""
fsdp_basics.py - Basic FSDP training example

Demonstrates:
- FSDP setup and configuration
- Different sharding strategies
- Mixed precision training
- Basic checkpointing

Usage:
    torchrun --nproc_per_node=4 fsdp_basics.py

Author: CUDA Lab
"""

import os
import time
import argparse
import functools
from typing import Optional

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
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model for FSDP demonstration."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        
        for layer in self.layers:
            x = layer(x)
        
        # Pool and classify
        x = x.mean(dim=1)
        return self.classifier(x)


def get_fsdp_config(
    sharding_strategy: str = "full_shard",
    use_mixed_precision: bool = True,
    cpu_offload: bool = False,
) -> dict:
    """Get FSDP configuration based on settings."""
    
    # Sharding strategy
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    sharding = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    # Mixed precision
    mixed_precision = None
    if use_mixed_precision:
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    
    # CPU offload
    offload = CPUOffload(offload_params=True) if cpu_offload else None
    
    # Auto wrap policy - wrap each TransformerBlock
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    
    return {
        "sharding_strategy": sharding,
        "mixed_precision": mixed_precision,
        "cpu_offload": offload,
        "auto_wrap_policy": auto_wrap_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "use_orig_params": True,  # Needed for torch.compile
    }


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size


def create_synthetic_data(num_samples: int = 10000, seq_len: int = 128, vocab_size: int = 10000):
    """Create synthetic data for testing."""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, 1000, (num_samples,))
    return TensorDataset(input_ids, labels)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return total_loss / num_batches


def get_memory_stats(device):
    """Get GPU memory statistics."""
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description="FSDP Training Example")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sharding", type=str, default="full_shard",
                        choices=["full_shard", "shard_grad_op", "no_shard", "hybrid_shard"])
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=768)
    args = parser.parse_args()
    
    # Setup distributed
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("=" * 60)
        print("FSDP Training Example")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Sharding strategy: {args.sharding}")
        print(f"Mixed precision: {not args.no_mixed_precision}")
        print(f"CPU offload: {args.cpu_offload}")
        print(f"Model: {args.num_layers} layers, d_model={args.d_model}")
        print("=" * 60)
    
    # Create model
    model = SimpleTransformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model parameters: {num_params / 1e6:.1f}M")
    
    # Get FSDP config
    fsdp_config = get_fsdp_config(
        sharding_strategy=args.sharding,
        use_mixed_precision=not args.no_mixed_precision,
        cpu_offload=args.cpu_offload,
    )
    
    # Wrap model with FSDP
    model = FSDP(model, **fsdp_config)
    
    if rank == 0:
        print(f"FSDP wrapped model")
        mem_stats = get_memory_stats(device)
        print(f"Memory after wrapping: {mem_stats['allocated_gb']:.2f} GB allocated")
    
    # Create data
    dataset = create_synthetic_data()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )
    
    # Optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    if rank == 0:
        print("\nStarting training...")
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, rank)
        epoch_time = time.time() - epoch_start
        
        if rank == 0:
            mem_stats = get_memory_stats(device)
            print(f"\nEpoch {epoch + 1}/{args.epochs}:")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Throughput: {len(dataset) / epoch_time:.0f} samples/s")
            print(f"  Peak Memory: {mem_stats['max_allocated_gb']:.2f} GB")
    
    # Save checkpoint example
    if rank == 0:
        print("\nSaving checkpoint...")
        
        # Save full state dict (for inference)
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            StateDictType,
        )
        
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
            torch.save(state_dict, "fsdp_model.pt")
            print("Saved checkpoint to fsdp_model.pt")
    
    # Final memory stats
    if rank == 0:
        print("\n" + "=" * 60)
        print("FINAL MEMORY STATISTICS")
        print("=" * 60)
        mem_stats = get_memory_stats(device)
        print(f"Allocated: {mem_stats['allocated_gb']:.2f} GB")
        print(f"Reserved: {mem_stats['reserved_gb']:.2f} GB")
        print(f"Peak: {mem_stats['max_allocated_gb']:.2f} GB")
        print("=" * 60)
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
