"""
checkpoint_utils.py - Reusable checkpointing utilities for PyTorch training

Features:
- Automatic checkpoint save/load
- Signal handling for preemption
- Best model tracking
- Distributed training support
- RNG state preservation for reproducibility
"""

import os
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import torch
import torch.distributed as dist


class CheckpointManager:
    """
    Manages checkpointing for PyTorch training with support for:
    - Regular interval checkpointing
    - Best model tracking
    - Preemption handling
    - Distributed training (DDP/FSDP)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        rank: int = 0,
        world_size: int = 1,
        save_best: bool = True,
        keep_last_n: int = 3,
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            model: PyTorch model (or DDP-wrapped model)
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            scaler: GradScaler for mixed precision (optional)
            rank: Process rank for distributed training
            world_size: Total number of processes
            save_best: Whether to track and save best model
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        self.save_best = save_best
        self.keep_last_n = keep_last_n
        
        self.best_metric = float('inf')  # Assuming lower is better
        self.current_epoch = 0
        self.global_step = 0
        
        # Register signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Register handlers for preemption signals."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGUSR1, self._signal_handler)
        
        if self.rank == 0:
            print("Checkpoint manager initialized. Signal handlers registered.")
    
    def _signal_handler(self, signum, frame):
        """Handle preemption signal by saving checkpoint."""
        print(f"[Rank {self.rank}] Received signal {signum}, saving checkpoint...")
        self.save('preempt_checkpoint.pt')
        sys.exit(0)
    
    def _get_model_state_dict(self) -> Dict[str, Any]:
        """Get model state dict, handling DDP wrapper."""
        if hasattr(self.model, 'module'):
            # DDP or DataParallel wrapped
            return self.model.module.state_dict()
        return self.model.state_dict()
    
    def _load_model_state_dict(self, state_dict: Dict[str, Any]):
        """Load model state dict, handling DDP wrapper."""
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
    
    def save(self, filename: str = 'checkpoint.pt', metrics: Optional[Dict] = None, 
             epoch: Optional[int] = None, loss: Optional[float] = None, **kwargs):
        """
        Save checkpoint. Only rank 0 saves in distributed setting.
        
        Args:
            filename: Checkpoint filename
            metrics: Optional metrics dict to include
            epoch: Optional epoch number (also updates self.current_epoch)
            loss: Optional loss value (added to metrics)
            **kwargs: Additional values to store in checkpoint
        """
        # Update epoch if provided
        if epoch is not None:
            self.current_epoch = epoch
            
        # Build metrics from loss if provided
        if metrics is None:
            metrics = {}
        if loss is not None:
            metrics['loss'] = loss
            
        # Only save on rank 0
        if self.rank != 0:
            if self.is_distributed:
                dist.barrier()
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self._get_model_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Save RNG states for reproducibility
        checkpoint['rng_states'] = {
            'python': torch.random.get_rng_state(),
            'numpy': None,  # Add if using numpy
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Sync in distributed setting
        if self.is_distributed:
            dist.barrier()
    
    def save_if_best(self, metric: float, filename: str = 'best_model.pt'):
        """Save if metric is better than previous best."""
        if metric < self.best_metric:  # Assuming lower is better
            self.best_metric = metric
            self.save(filename)
            if self.rank == 0:
                print(f"New best metric: {metric:.6f}")
            return True
        return False
    
    def load(self, filename: str = 'checkpoint.pt', strict: bool = True) -> bool:
        """
        Load checkpoint.
        
        Args:
            filename: Checkpoint filename
            strict: Strict mode for state dict loading
            
        Returns:
            True if checkpoint was loaded, False otherwise
        """
        path = self.checkpoint_dir / filename
        
        if not path.exists():
            if self.rank == 0:
                print(f"No checkpoint found at {path}")
            return False
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load model state
        self._load_model_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        # Restore RNG states
        if 'rng_states' in checkpoint:
            rng = checkpoint['rng_states']
            if rng.get('python') is not None:
                torch.random.set_rng_state(rng['python'])
            if rng.get('cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng['cuda'])
        
        if self.rank == 0:
            print(f"Loaded checkpoint from {path}")
            print(f"  Epoch: {self.current_epoch}, Step: {self.global_step}")
            print(f"  Best metric: {self.best_metric:.6f}")
        
        return True
    
    def load_best(self) -> bool:
        """Load the best model checkpoint."""
        return self.load('best_model.pt')
    
    def load_latest(self) -> Optional[int]:
        """
        Load the most recent checkpoint.
        
        Returns:
            The epoch number if loaded successfully, None otherwise.
        """
        if self.auto_resume():
            return self.current_epoch
        return None

    def auto_resume(self) -> bool:
        """
        Attempt to resume from latest checkpoint.
        Checks for: preempt_checkpoint.pt, checkpoint.pt, latest_*.pt
        """
        # Priority order for resuming
        candidates = [
            'preempt_checkpoint.pt',  # Highest priority: preemption save
            'checkpoint.pt',          # Regular checkpoint
        ]
        
        # Also check for numbered checkpoints
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pt'))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            candidates.append(latest.name)
        
        for candidate in candidates:
            if self.load(candidate):
                return True
        
        return False
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if self.keep_last_n <= 0:
            return
        
        # Find numbered checkpoints
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pt'))
        
        if len(checkpoints) > self.keep_last_n:
            # Sort by modification time
            checkpoints.sort(key=lambda p: p.stat().st_mtime)
            
            # Remove oldest
            for ckpt in checkpoints[:-self.keep_last_n]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt}")
    
    def step(self, epoch: Optional[int] = None, global_step: Optional[int] = None):
        """Update current training state."""
        if epoch is not None:
            self.current_epoch = epoch
        if global_step is not None:
            self.global_step = global_step


# Convenience function for simple use cases
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    scheduler: Optional[Any] = None,
    **kwargs
):
    """Simple checkpoint save function."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    scheduler: Optional[Any] = None,
) -> int:
    """Simple checkpoint load function. Returns epoch."""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Loaded checkpoint from {path}")
    return checkpoint.get('epoch', 0)
