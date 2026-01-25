#!/usr/bin/env python3
"""
elastic_training.py - Elastic training with fault tolerance

Features:
- Elastic scaling (add/remove nodes during training)
- Automatic recovery from node failures
- State preservation across restarts
- Integration with torchrun elastic

Usage:
    # Launch with torchrun elastic
    torchrun \
        --nnodes=1:4 \
        --nproc_per_node=4 \
        --rdzv_id=job_123 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=master:29400 \
        elastic_training.py

Author: CUDA Lab
"""

import os
import socket
import signal
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.elastic.multiprocessing.errors import record


@dataclass
class ElasticConfig:
    """Configuration for elastic training."""
    
    # Checkpoint settings
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 100  # Save every N steps
    
    # Recovery settings
    max_restarts: int = 3
    restart_interval: int = 5  # Seconds between restarts
    
    # Training settings
    min_nodes: int = 1
    max_nodes: int = 4
    
    # Timeout settings
    collective_timeout: int = 300  # 5 minutes


class ElasticState:
    """
    Maintains elastic training state across restarts.
    
    State is preserved when:
    - Nodes join/leave the job
    - Checkpoints are saved/loaded
    - Process restarts after failure
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state
        self.current_step: int = 0
        self.current_epoch: int = 0
        self.best_loss: float = float("inf")
        
        # Elastic state
        self.num_restarts: int = 0
        self.world_history: list = []
    
    def save(self, model: nn.Module, optimizer, scheduler=None):
        """Save state to checkpoint."""
        checkpoint = {
            "step": self.current_step,
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "num_restarts": self.num_restarts,
            "world_history": self.world_history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save with rank in filename
        rank = dist.get_rank() if dist.is_initialized() else 0
        path = os.path.join(self.checkpoint_dir, f"elastic_ckpt_rank{rank}.pt")
        
        # Atomic save
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.rename(tmp_path, path)
        
        if rank == 0:
            print(f"Saved elastic checkpoint at step {self.current_step}")
    
    def load(self, model: nn.Module, optimizer, scheduler=None) -> bool:
        """Load state from checkpoint. Returns True if loaded."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        path = os.path.join(self.checkpoint_dir, f"elastic_ckpt_rank{rank}.pt")
        
        if not os.path.exists(path):
            # Try rank 0 checkpoint
            path = os.path.join(self.checkpoint_dir, "elastic_ckpt_rank0.pt")
            if not os.path.exists(path):
                return False
        
        checkpoint = torch.load(path, map_location="cpu")
        
        self.current_step = checkpoint["step"]
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.num_restarts = checkpoint.get("num_restarts", 0)
        self.world_history = checkpoint.get("world_history", [])
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.num_restarts += 1
        
        if rank == 0:
            print(f"Loaded checkpoint from step {self.current_step}")
            print(f"This is restart #{self.num_restarts}")
        
        return True
    
    def record_world_change(self):
        """Record a change in the distributed world."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.world_history.append({
            "step": self.current_step,
            "world_size": world_size,
            "timestamp": time.time(),
        })


class ElasticTrainer:
    """
    Trainer with elastic training support.
    
    Handles:
    - Node failures and restarts
    - Dynamic node addition/removal
    - State preservation
    - Proper cleanup
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        dataloader: DataLoader,
        config: ElasticConfig,
        scheduler=None,
        loss_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Initialize distributed if not already
        if not dist.is_initialized():
            self._init_distributed()
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        # Move model to device and wrap with DDP
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank])
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        
        # Elastic state
        self.state = ElasticState(config.checkpoint_dir)
        
        # Try to resume from checkpoint
        if self._try_resume():
            print(f"Rank {self.rank}: Resumed from step {self.state.current_step}")
        
        # Record initial world
        self.state.record_world_change()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _init_distributed(self):
        """Initialize distributed training."""
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(
            seconds=self.config.collective_timeout
        ))
    
    def _try_resume(self) -> bool:
        """Try to resume from checkpoint."""
        return self.state.load(
            self.model.module if hasattr(self.model, "module") else self.model,
            self.optimizer,
            self.scheduler
        )
    
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def handler(signum, frame):
            print(f"Rank {self.rank}: Received signal {signum}, saving checkpoint...")
            self._save_checkpoint()
            raise SystemExit(0)
        
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        self.state.save(model, self.optimizer, self.scheduler)
    
    def _detect_world_change(self) -> bool:
        """Detect if world size has changed."""
        current_world_size = dist.get_world_size()
        if self.world_size != current_world_size:
            if self.rank == 0:
                print(f"World size changed: {self.world_size} -> {current_world_size}")
            self.world_size = current_world_size
            self.state.record_world_change()
            return True
        return False
    
    def train_step(self, batch) -> float:
        """Execute a single training step."""
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self):
        """Train for one epoch with elastic support."""
        self.model.train()
        
        # Update sampler for new epoch
        if hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(self.state.current_epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Skip batches if resuming mid-epoch
            if batch_idx < self.state.current_step % len(self.dataloader):
                continue
            
            try:
                loss = self.train_step(batch)
                total_loss += loss
                num_batches += 1
                self.state.current_step += 1
                
                # Periodic checkpoint
                if self.state.current_step % self.config.save_interval == 0:
                    self._save_checkpoint()
                
                # Check for world changes
                self._detect_world_change()
                
                # Log progress
                if self.rank == 0 and batch_idx % 10 == 0:
                    print(
                        f"Epoch {self.state.current_epoch} | "
                        f"Step {self.state.current_step} | "
                        f"Loss: {loss:.4f}"
                    )
                
            except RuntimeError as e:
                # Handle NCCL errors gracefully
                if "NCCL" in str(e):
                    print(f"Rank {self.rank}: NCCL error, saving checkpoint...")
                    self._save_checkpoint()
                    raise
                raise
        
        self.state.current_epoch += 1
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
        
        return total_loss / max(num_batches, 1)
    
    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        start_epoch = self.state.current_epoch
        
        for epoch in range(start_epoch, num_epochs):
            if self.rank == 0:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"World size: {self.world_size}")
                print(f"{'='*50}")
            
            avg_loss = self.train_epoch()
            
            if self.rank == 0:
                print(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self._save_checkpoint()
            
            if self.scheduler:
                self.scheduler.step()
        
        if self.rank == 0:
            print("\nTraining complete!")
            print(f"Best loss: {self.state.best_loss:.4f}")
            print(f"Total restarts: {self.state.num_restarts}")


# ============================================================================
# Elastic Launch Utilities
# ============================================================================

def get_elastic_info() -> Dict[str, Any]:
    """Get information about the current elastic job."""
    return {
        "rank": int(os.environ.get("RANK", 0)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "group_rank": int(os.environ.get("GROUP_RANK", 0)),
        "role_rank": int(os.environ.get("ROLE_RANK", 0)),
        "role_world_size": int(os.environ.get("ROLE_WORLD_SIZE", 1)),
        "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
        "master_port": os.environ.get("MASTER_PORT", "29500"),
        "restart_count": int(os.environ.get("TORCHELASTIC_RESTART_COUNT", 0)),
        "max_restarts": int(os.environ.get("TORCHELASTIC_MAX_RESTARTS", 0)),
        "hostname": socket.gethostname(),
    }


def print_elastic_info():
    """Print elastic training information."""
    info = get_elastic_info()
    
    print("\n" + "=" * 50)
    print("ELASTIC TRAINING ENVIRONMENT")
    print("=" * 50)
    
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("=" * 50 + "\n")


# ============================================================================
# Demo with @record decorator
# ============================================================================

@record  # This decorator helps with error recording
def main():
    """
    Demo elastic training.
    
    Run with:
        torchrun --nnodes=1:4 --nproc_per_node=2 \
            --rdzv_id=job_${JOB_ID} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=localhost:29400 \
            elastic_training.py
    """
    import datetime
    import argparse
    from torch.utils.data import TensorDataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    # Print environment info
    print_elastic_info()
    
    # Initialize distributed
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=300),
    )
    
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Create dummy data
    X = torch.randn(1000, 10)
    y = torch.randint(0, 5, (1000,))
    dataset = TensorDataset(X, y)
    
    # Create sampler and dataloader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=True,
    )
    
    # Create model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 5),
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create config
    config = ElasticConfig(
        checkpoint_dir=args.checkpoint_dir,
        save_interval=50,
    )
    
    # Create trainer
    trainer = ElasticTrainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        config=config,
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
