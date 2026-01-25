#!/usr/bin/env python3
"""
checkpoint_manager.py - Robust checkpointing for distributed training

Features:
- Automatic checkpoint saving at intervals
- Keep last N checkpoints
- Save best model based on metric
- Async checkpoint saving
- Resume from latest checkpoint
- Support for FSDP/DeepSpeed checkpoints

Usage:
    manager = CheckpointManager(
        checkpoint_dir="/path/to/checkpoints",
        keep_last_n=3,
        save_best=True,
    )
    
    # Save
    manager.save(model, optimizer, step, metrics={"loss": 0.5})
    
    # Load
    state = manager.load_latest(model, optimizer)

Author: CUDA Lab
"""

import os
import re
import json
import shutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint manager."""
    checkpoint_dir: str = "./checkpoints"
    
    # Saving strategy
    save_interval_steps: int = 1000
    save_interval_minutes: int = 30
    keep_last_n: int = 3
    save_best: bool = True
    best_metric: str = "loss"
    best_mode: str = "min"  # "min" or "max"
    
    # Performance
    async_save: bool = True
    
    # FSDP support
    use_fsdp: bool = False
    fsdp_save_full: bool = True


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    step: int
    epoch: int
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    is_best: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "is_best": self.is_best,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "CheckpointMetadata":
        return cls(**d)


class CheckpointManager:
    """
    Manages training checkpoints with automatic cleanup and best model tracking.
    
    Features:
    - Save checkpoints at regular intervals
    - Keep only last N checkpoints
    - Track and save best model
    - Async saving to not block training
    - Resume from latest or best checkpoint
    """
    
    def __init__(
        self,
        config: Optional[CheckpointConfig] = None,
        checkpoint_dir: Optional[str] = None,
        keep_last_n: int = 3,
        save_best: bool = True,
        best_metric: str = "loss",
        best_mode: str = "min",
        rank: int = 0,
    ):
        self.config = config or CheckpointConfig()
        
        # Override config with explicit args
        if checkpoint_dir:
            self.config.checkpoint_dir = checkpoint_dir
        if keep_last_n:
            self.config.keep_last_n = keep_last_n
        self.config.save_best = save_best
        self.config.best_metric = best_metric
        self.config.best_mode = best_mode
        
        self.rank = rank
        self.is_main = rank == 0
        
        # State
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_metric_value: Optional[float] = None
        self.last_save_time = datetime.now()
        self.last_save_step = 0
        
        # Async saving
        self.save_thread: Optional[threading.Thread] = None
        
        # Create checkpoint directory
        if self.is_main:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """Load metadata of existing checkpoints."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        # Find all checkpoint directories
        for item in checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_"):
                metadata_path = item / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = CheckpointMetadata.from_dict(json.load(f))
                        self.checkpoints.append(metadata)
        
        # Sort by step
        self.checkpoints.sort(key=lambda x: x.step)
        
        # Find best metric value
        if self.config.save_best:
            for ckpt in self.checkpoints:
                if self.config.best_metric in ckpt.metrics:
                    value = ckpt.metrics[self.config.best_metric]
                    if self.best_metric_value is None:
                        self.best_metric_value = value
                    elif self._is_better(value, self.best_metric_value):
                        self.best_metric_value = value
    
    def _is_better(self, new: float, current: float) -> bool:
        """Check if new metric is better than current."""
        if self.config.best_mode == "min":
            return new < current
        return new > current
    
    def _get_checkpoint_path(self, step: int) -> str:
        """Get checkpoint directory path for a step."""
        return os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{step:08d}"
        )
    
    def should_save(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step."""
        # Check step interval
        if step - self.last_save_step >= self.config.save_interval_steps:
            return True
        
        # Check time interval
        elapsed = (datetime.now() - self.last_save_time).total_seconds() / 60
        if elapsed >= self.config.save_interval_minutes:
            return True
        
        return False
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict] = None,
    ):
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer
            scheduler: Optional LR scheduler
            step: Current training step
            epoch: Current epoch
            metrics: Dictionary of metrics
            extra_state: Additional state to save
        """
        if not self.is_main:
            # Non-main ranks wait at barrier
            if dist.is_initialized():
                dist.barrier()
            return
        
        metrics = metrics or {}
        
        # Check if this is the best model
        is_best = False
        if self.config.save_best and self.config.best_metric in metrics:
            value = metrics[self.config.best_metric]
            if self.best_metric_value is None or self._is_better(value, self.best_metric_value):
                self.best_metric_value = value
                is_best = True
        
        # Create metadata
        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            is_best=is_best,
        )
        
        # Prepare checkpoint path
        checkpoint_path = self._get_checkpoint_path(step)
        
        # Save synchronously or asynchronously
        if self.config.async_save:
            # Wait for previous save to complete
            if self.save_thread and self.save_thread.is_alive():
                self.save_thread.join()
            
            # Start async save
            self.save_thread = threading.Thread(
                target=self._save_checkpoint,
                args=(model, optimizer, scheduler, checkpoint_path, metadata, extra_state),
            )
            self.save_thread.start()
            
            # If this is the best model, wait for save to complete before copying
            if is_best:
                self.save_thread.join()
        else:
            self._save_checkpoint(model, optimizer, scheduler, checkpoint_path, metadata, extra_state)
        
        # Update state
        self.checkpoints.append(metadata)
        self.last_save_time = datetime.now()
        self.last_save_step = step
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save best model
        if is_best:
            self._save_best(checkpoint_path)
        
        # Barrier for distributed
        if dist.is_initialized():
            dist.barrier()
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer],
        scheduler: Optional[Any],
        checkpoint_path: str,
        metadata: CheckpointMetadata,
        extra_state: Optional[Dict],
    ):
        """Actually save the checkpoint to disk."""
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Prepare state dict
        state = {
            "model_state_dict": model.state_dict(),
            "step": metadata.step,
            "epoch": metadata.epoch,
        }
        
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
        
        if scheduler:
            state["scheduler_state_dict"] = scheduler.state_dict()
        
        if extra_state:
            state.update(extra_state)
        
        # Save model
        model_path = os.path.join(checkpoint_path, "model.pt")
        torch.save(state, model_path)
        
        # Save metadata
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_best(self, checkpoint_path: str):
        """Copy checkpoint to best model location."""
        best_path = os.path.join(self.config.checkpoint_dir, "best")
        
        # Remove existing best
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        
        # Copy new best
        shutil.copytree(checkpoint_path, best_path)
        print(f"Saved best model to {best_path}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only last N."""
        if len(self.checkpoints) <= self.config.keep_last_n:
            return
        
        # Sort by step, keep latest
        sorted_ckpts = sorted(self.checkpoints, key=lambda x: x.step)
        to_remove = sorted_ckpts[:-self.config.keep_last_n]
        
        for ckpt in to_remove:
            path = self._get_checkpoint_path(ckpt.step)
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"Removed old checkpoint: {path}")
            self.checkpoints.remove(ckpt)
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Optional[Dict]:
        """Load the latest checkpoint."""
        if not self.checkpoints:
            print("No checkpoints found")
            return None
        
        latest = max(self.checkpoints, key=lambda x: x.step)
        return self.load(
            self._get_checkpoint_path(latest.step),
            model, optimizer, scheduler
        )
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Optional[Dict]:
        """Load the best checkpoint."""
        best_path = os.path.join(self.config.checkpoint_dir, "best")
        
        if not os.path.exists(best_path):
            print("No best checkpoint found")
            return None
        
        return self.load(best_path, model, optimizer, scheduler)
    
    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict:
        """Load a specific checkpoint."""
        model_path = os.path.join(checkpoint_path, "model.pt")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(model_path, map_location="cpu")
        
        # Load model
        model.load_state_dict(state["model_state_dict"])
        
        # Load optimizer
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        
        # Load scheduler
        if scheduler and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        
        return state
    
    def wait_for_save(self):
        """Wait for async save to complete."""
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join()


# ============================================================================
# FSDP Checkpoint Support
# ============================================================================

class FSDPCheckpointManager(CheckpointManager):
    """
    Checkpoint manager with FSDP support.
    
    Handles:
    - Sharded state dict (fast, requires same world size)
    - Full state dict (slow, portable)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.use_fsdp = True
    
    def save(
        self,
        model: nn.Module,  # FSDP wrapped model
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict] = None,
    ):
        """Save FSDP checkpoint."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
            ShardedStateDictConfig,
        )
        
        metrics = metrics or {}
        
        # Determine if best
        is_best = False
        if self.config.save_best and self.config.best_metric in metrics:
            value = metrics[self.config.best_metric]
            if self.best_metric_value is None or self._is_better(value, self.best_metric_value):
                self.best_metric_value = value
                is_best = True
        
        checkpoint_path = self._get_checkpoint_path(step)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save sharded state for resumption
        sharded_config = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_config):
            state_dict = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer) if optimizer else None
        
        # Each rank saves its shard
        rank = dist.get_rank() if dist.is_initialized() else 0
        shard_path = os.path.join(checkpoint_path, f"shard_rank{rank}.pt")
        torch.save({
            "model_state_dict": state_dict,
            "optimizer_state_dict": optim_state,
            "step": step,
            "epoch": epoch,
        }, shard_path)
        
        # Save full state dict for inference (optional, rank 0 only)
        if self.config.fsdp_save_full and self.is_main:
            full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_config):
                full_state = model.state_dict()
            
            full_path = os.path.join(checkpoint_path, "model_full.pt")
            torch.save(full_state, full_path)
        
        # Save metadata (rank 0 only)
        if self.is_main:
            metadata = CheckpointMetadata(
                step=step,
                epoch=epoch,
                timestamp=datetime.now().isoformat(),
                metrics=metrics,
                is_best=is_best,
            )
            
            metadata_path = os.path.join(checkpoint_path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            self.checkpoints.append(metadata)
            self.last_save_time = datetime.now()
            self.last_save_step = step
            
            self._cleanup_old_checkpoints()
            
            if is_best:
                self._save_best(checkpoint_path)
            
            print(f"Saved FSDP checkpoint to {checkpoint_path}")
        
        dist.barrier()


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate checkpoint manager."""
    import tempfile
    
    print("=" * 60)
    print("CHECKPOINT MANAGER DEMO")
    print("=" * 60)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create manager
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            checkpoint_dir=tmpdir,
            keep_last_n=3,
            save_best=True,
            best_metric="loss",
            best_mode="min",
        )
        
        # Simulate training
        print("\nSimulating training...")
        
        for step in range(1, 6):
            # Fake training step
            loss = 1.0 / step
            
            # Save checkpoint
            print(f"\nStep {step}: loss={loss:.4f}")
            manager.save(
                model=model,
                optimizer=optimizer,
                step=step * 1000,
                epoch=step,
                metrics={"loss": loss, "accuracy": step * 0.2},
            )
        
        # List checkpoints
        print("\nCheckpoints:")
        for ckpt in manager.checkpoints:
            print(f"  Step {ckpt.step}: loss={ckpt.metrics.get('loss', 'N/A'):.4f}, best={ckpt.is_best}")
        
        # Load latest
        print("\nLoading latest checkpoint...")
        state = manager.load_latest(model, optimizer)
        print(f"Loaded step {state['step']}, epoch {state['epoch']}")
        
        # Load best
        print("\nLoading best checkpoint...")
        state = manager.load_best(model, optimizer)
        print(f"Loaded step {state['step']}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
