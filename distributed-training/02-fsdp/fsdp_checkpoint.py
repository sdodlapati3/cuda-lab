#!/usr/bin/env python3
"""
fsdp_checkpoint.py - Advanced FSDP checkpointing strategies

Demonstrates:
- Full state dict checkpointing (for inference)
- Sharded state dict checkpointing (for training resumption)
- Distributed checkpoint with torch.distributed.checkpoint
- Efficient checkpoint loading and resumption
- Converting between checkpoint formats

Usage:
    torchrun --nproc_per_node=4 fsdp_checkpoint.py --mode save
    torchrun --nproc_per_node=4 fsdp_checkpoint.py --mode load
    torchrun --nproc_per_node=4 fsdp_checkpoint.py --mode convert

Author: CUDA Lab
"""

import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
    StateDictType,
    FullStateDictConfig,
    ShardedStateDictConfig,
    LocalStateDictConfig,
    FullOptimStateDictConfig,
    ShardedOptimStateDictConfig,
)

# Distributed checkpoint (torch 2.0+)
try:
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict,
        set_state_dict,
        StateDictOptions,
    )
    DIST_CHECKPOINT_AVAILABLE = True
except ImportError:
    DIST_CHECKPOINT_AVAILABLE = False
    print("Warning: torch.distributed.checkpoint not available")


# Simple model for demonstration
class TransformerBlock(nn.Module):
    """Transformer block for FSDP wrapping."""
    
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_ff: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class SimpleTransformer(nn.Module):
    """Simple transformer for checkpoint demonstration."""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 512, num_layers: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(dim=1))


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def create_fsdp_model(d_model: int = 512, num_layers: int = 6) -> FSDP:
    """Create FSDP-wrapped model."""
    import functools
    
    model = SimpleTransformer(d_model=d_model, num_layers=num_layers)
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )
    
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
    )
    
    return model


# ==============================================================================
# Checkpoint Strategies
# ==============================================================================

class FullStateDictCheckpoint:
    """
    Full State Dict Checkpoint
    
    Pros:
    - Compatible with single-GPU inference
    - Can be loaded without FSDP
    - Standard PyTorch checkpoint format
    
    Cons:
    - Requires gathering all shards to rank 0
    - High memory usage on rank 0
    - Slow for very large models
    
    Best for:
    - Final model export for inference
    - Converting to other formats (GGUF, etc.)
    """
    
    @staticmethod
    def save(
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer],
        path: str,
        rank: int,
        extra_state: Optional[Dict] = None,
    ):
        """Save full state dict checkpoint (only rank 0 saves)."""
        save_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )
        optim_save_policy = FullOptimStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy, optim_save_policy):
            state_dict = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer) if optimizer else None
        
        if rank == 0:
            checkpoint = {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optim_state,
            }
            if extra_state:
                checkpoint.update(extra_state)
            
            torch.save(checkpoint, path)
            print(f"Saved full state dict checkpoint to {path}")
    
    @staticmethod
    def load(
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer],
        path: str,
        rank: int,
    ) -> Dict:
        """Load full state dict checkpoint."""
        # Load on rank 0 and broadcast
        if rank == 0:
            checkpoint = torch.load(path, map_location="cpu")
        else:
            checkpoint = None
        
        # Broadcast checkpoint to all ranks
        checkpoint = [checkpoint]
        dist.broadcast_object_list(checkpoint, src=0)
        checkpoint = checkpoint[0]
        
        # Load model state
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optim_state = FSDP.optim_state_dict_to_load(
                model, optimizer, checkpoint["optimizer_state_dict"]
            )
            optimizer.load_state_dict(optim_state)
        
        return checkpoint


class ShardedStateDictCheckpoint:
    """
    Sharded State Dict Checkpoint
    
    Pros:
    - Each rank saves its own shard
    - Lower peak memory usage
    - Faster save/load for large models
    
    Cons:
    - Requires same world size to resume
    - Cannot be used for single-GPU inference directly
    
    Best for:
    - Training checkpoints for resumption
    - When memory is constrained
    """
    
    @staticmethod
    def save(
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer],
        checkpoint_dir: str,
        rank: int,
        extra_state: Optional[Dict] = None,
    ):
        """Save sharded checkpoint (each rank saves its shard)."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        sharded_policy = ShardedStateDictConfig(offload_to_cpu=True)
        sharded_optim_policy = ShardedOptimStateDictConfig(offload_to_cpu=True)
        
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_policy, sharded_optim_policy):
            state_dict = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer) if optimizer else None
        
        checkpoint = {
            "model_state_dict": state_dict,
            "optimizer_state_dict": optim_state,
        }
        if extra_state:
            checkpoint.update(extra_state)
        
        # Each rank saves its shard
        shard_path = os.path.join(checkpoint_dir, f"shard_rank{rank}.pt")
        torch.save(checkpoint, shard_path)
        
        dist.barrier()
        if rank == 0:
            print(f"Saved sharded checkpoint to {checkpoint_dir}")
    
    @staticmethod
    def load(
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer],
        checkpoint_dir: str,
        rank: int,
    ) -> Dict:
        """Load sharded checkpoint."""
        shard_path = os.path.join(checkpoint_dir, f"shard_rank{rank}.pt")
        checkpoint = torch.load(shard_path, map_location="cpu")
        
        sharded_policy = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_policy):
            model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optim_state = FSDP.optim_state_dict_to_load(
                model, optimizer, checkpoint["optimizer_state_dict"]
            )
            optimizer.load_state_dict(optim_state)
        
        dist.barrier()
        return checkpoint


class DistributedCheckpoint:
    """
    Distributed Checkpoint (torch.distributed.checkpoint)
    
    Pros:
    - World-size agnostic (can resume with different # of GPUs)
    - Efficient parallel I/O
    - Supports async save
    
    Cons:
    - Requires torch 2.0+
    - More complex setup
    
    Best for:
    - Production training with elastic scaling
    - Very large models
    """
    
    @staticmethod
    def save(
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer],
        checkpoint_dir: str,
        rank: int,
        extra_state: Optional[Dict] = None,
    ):
        """Save using distributed checkpoint."""
        if not DIST_CHECKPOINT_AVAILABLE:
            raise RuntimeError("torch.distributed.checkpoint not available")
        
        # Clear existing checkpoint
        if rank == 0 and os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        dist.barrier()
        
        # Get state dicts using new API
        model_state, optimizer_state = get_state_dict(
            model, optimizer,
            options=StateDictOptions(
                full_state_dict=False,
                cpu_offload=True,
            )
        )
        
        state_dict = {
            "model": model_state,
            "optimizer": optimizer_state,
        }
        if extra_state:
            state_dict.update(extra_state)
        
        # Save with distributed checkpoint
        dist_cp.save(state_dict, checkpoint_dir=checkpoint_dir)
        
        if rank == 0:
            print(f"Saved distributed checkpoint to {checkpoint_dir}")
    
    @staticmethod
    def load(
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer],
        checkpoint_dir: str,
        rank: int,
    ) -> Dict:
        """Load using distributed checkpoint."""
        if not DIST_CHECKPOINT_AVAILABLE:
            raise RuntimeError("torch.distributed.checkpoint not available")
        
        # Get current state dict structure
        model_state, optimizer_state = get_state_dict(
            model, optimizer,
            options=StateDictOptions(full_state_dict=False)
        )
        
        state_dict = {
            "model": model_state,
            "optimizer": optimizer_state,
        }
        
        # Load checkpoint
        dist_cp.load(state_dict, checkpoint_dir=checkpoint_dir)
        
        # Set state back to model/optimizer
        set_state_dict(
            model, optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
            options=StateDictOptions(full_state_dict=False),
        )
        
        return state_dict


# ==============================================================================
# Checkpoint Conversion
# ==============================================================================

def convert_sharded_to_full(
    model: FSDP,
    sharded_dir: str,
    output_path: str,
    rank: int,
):
    """Convert sharded checkpoint to full state dict for inference."""
    # Load sharded checkpoint
    shard_path = os.path.join(sharded_dir, f"shard_rank{rank}.pt")
    checkpoint = torch.load(shard_path, map_location="cpu")
    
    sharded_policy = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_policy):
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Now save as full state dict
    full_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_policy):
        full_state_dict = model.state_dict()
    
    if rank == 0:
        torch.save({"model_state_dict": full_state_dict}, output_path)
        print(f"Converted sharded checkpoint to full state dict: {output_path}")


# ==============================================================================
# Demo Functions
# ==============================================================================

def demo_save_checkpoints(rank: int, world_size: int):
    """Demonstrate saving checkpoints with different strategies."""
    print(f"\n{'='*60}")
    print("DEMONSTRATING CHECKPOINT SAVE STRATEGIES")
    print(f"{'='*60}\n")
    
    # Create model and optimizer
    model = create_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Simulate some training
    dummy_input = torch.randint(0, 10000, (4, 128)).cuda()
    for _ in range(5):
        output = model(dummy_input)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    extra_state = {"epoch": 1, "global_step": 100}
    
    # Strategy 1: Full State Dict
    if rank == 0:
        print("\n1. Saving Full State Dict Checkpoint...")
    start = time.time()
    FullStateDictCheckpoint.save(
        model, optimizer, "checkpoints/full_checkpoint.pt", rank, extra_state
    )
    dist.barrier()
    if rank == 0:
        print(f"   Time: {time.time() - start:.2f}s")
    
    # Strategy 2: Sharded State Dict
    if rank == 0:
        print("\n2. Saving Sharded State Dict Checkpoint...")
    start = time.time()
    ShardedStateDictCheckpoint.save(
        model, optimizer, "checkpoints/sharded_checkpoint", rank, extra_state
    )
    dist.barrier()
    if rank == 0:
        print(f"   Time: {time.time() - start:.2f}s")
    
    # Strategy 3: Distributed Checkpoint
    if DIST_CHECKPOINT_AVAILABLE:
        if rank == 0:
            print("\n3. Saving Distributed Checkpoint...")
        start = time.time()
        DistributedCheckpoint.save(
            model, optimizer, "checkpoints/dist_checkpoint", rank, extra_state
        )
        dist.barrier()
        if rank == 0:
            print(f"   Time: {time.time() - start:.2f}s")
    
    if rank == 0:
        print("\n✓ All checkpoints saved successfully!")
        print("\nCheckpoint sizes:")
        for path in ["checkpoints/full_checkpoint.pt"]:
            if os.path.exists(path):
                size = os.path.getsize(path) / 1e6
                print(f"  {path}: {size:.1f} MB")
        
        sharded_dir = "checkpoints/sharded_checkpoint"
        if os.path.exists(sharded_dir):
            total_size = sum(
                os.path.getsize(os.path.join(sharded_dir, f))
                for f in os.listdir(sharded_dir)
            ) / 1e6
            print(f"  {sharded_dir}/: {total_size:.1f} MB total")


def demo_load_checkpoints(rank: int, world_size: int):
    """Demonstrate loading checkpoints with different strategies."""
    print(f"\n{'='*60}")
    print("DEMONSTRATING CHECKPOINT LOAD STRATEGIES")
    print(f"{'='*60}\n")
    
    # Strategy 1: Full State Dict
    if os.path.exists("checkpoints/full_checkpoint.pt"):
        if rank == 0:
            print("\n1. Loading Full State Dict Checkpoint...")
        model = create_fsdp_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        start = time.time()
        state = FullStateDictCheckpoint.load(
            model, optimizer, "checkpoints/full_checkpoint.pt", rank
        )
        dist.barrier()
        if rank == 0:
            print(f"   Loaded epoch: {state.get('epoch')}, step: {state.get('global_step')}")
            print(f"   Time: {time.time() - start:.2f}s")
    
    # Strategy 2: Sharded State Dict
    if os.path.exists("checkpoints/sharded_checkpoint"):
        if rank == 0:
            print("\n2. Loading Sharded State Dict Checkpoint...")
        model = create_fsdp_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        start = time.time()
        state = ShardedStateDictCheckpoint.load(
            model, optimizer, "checkpoints/sharded_checkpoint", rank
        )
        dist.barrier()
        if rank == 0:
            print(f"   Loaded epoch: {state.get('epoch')}, step: {state.get('global_step')}")
            print(f"   Time: {time.time() - start:.2f}s")
    
    # Strategy 3: Distributed Checkpoint
    if DIST_CHECKPOINT_AVAILABLE and os.path.exists("checkpoints/dist_checkpoint"):
        if rank == 0:
            print("\n3. Loading Distributed Checkpoint...")
        model = create_fsdp_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        start = time.time()
        DistributedCheckpoint.load(
            model, optimizer, "checkpoints/dist_checkpoint", rank
        )
        dist.barrier()
        if rank == 0:
            print(f"   Time: {time.time() - start:.2f}s")
    
    if rank == 0:
        print("\n✓ All checkpoints loaded successfully!")


def demo_convert_checkpoint(rank: int, world_size: int):
    """Demonstrate checkpoint format conversion."""
    print(f"\n{'='*60}")
    print("DEMONSTRATING CHECKPOINT CONVERSION")
    print(f"{'='*60}\n")
    
    if not os.path.exists("checkpoints/sharded_checkpoint"):
        if rank == 0:
            print("No sharded checkpoint found. Run with --mode save first.")
        return
    
    if rank == 0:
        print("Converting sharded checkpoint to full state dict...")
    
    model = create_fsdp_model()
    convert_sharded_to_full(
        model,
        "checkpoints/sharded_checkpoint",
        "checkpoints/converted_full.pt",
        rank,
    )
    
    if rank == 0:
        print("\n✓ Conversion completed!")
        if os.path.exists("checkpoints/converted_full.pt"):
            size = os.path.getsize("checkpoints/converted_full.pt") / 1e6
            print(f"  Output size: {size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="FSDP Checkpoint Strategies")
    parser.add_argument("--mode", type=str, default="save",
                        choices=["save", "load", "convert", "all"])
    args = parser.parse_args()
    
    local_rank, rank, world_size = setup_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("FSDP CHECKPOINTING DEMO")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Mode: {args.mode}")
        print(f"Distributed checkpoint available: {DIST_CHECKPOINT_AVAILABLE}")
    
    try:
        if args.mode == "save" or args.mode == "all":
            demo_save_checkpoints(rank, world_size)
        
        if args.mode == "load" or args.mode == "all":
            demo_load_checkpoints(rank, world_size)
        
        if args.mode == "convert" or args.mode == "all":
            demo_convert_checkpoint(rank, world_size)
    
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
