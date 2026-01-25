#!/usr/bin/env python3
"""
ddp_training.py - Production-ready DDP training template

A comprehensive template for distributed training with:
- Gradient accumulation
- Mixed precision training
- Checkpointing and resume
- Learning rate scheduling
- Logging and metrics
- NVTX annotations for profiling

Usage:
    torchrun --nproc_per_node=4 ddp_training.py --config config.yaml

Author: CUDA Lab
"""

import os
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# Optional: NVTX for profiling
try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "resnet50"
    num_classes: int = 1000
    
    # Data
    data_dir: str = "./data"
    batch_size: int = 32  # Per GPU
    num_workers: int = 4
    
    # Training
    epochs: int = 100
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Gradient accumulation
    accumulation_steps: int = 1
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5  # Save every N epochs
    resume_from: Optional[str] = None
    
    # Logging
    log_every: int = 100  # Log every N steps
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size across all GPUs and accumulation steps."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return self.batch_size * world_size * self.accumulation_steps


class DistributedTrainer:
    """Production-ready distributed trainer."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_distributed()
        self.setup_logging()
        
    def setup_distributed(self):
        """Initialize distributed training environment."""
        dist.init_process_group(backend='nccl')
        
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        if self.rank == 0:
            print(f"Initialized distributed training: {self.world_size} GPUs")
            print(f"Effective batch size: {self.config.effective_batch_size}")
    
    def setup_logging(self):
        """Setup logging (only on rank 0)."""
        if self.rank == 0:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler('training.log')
                ]
            )
        else:
            logging.basicConfig(level=logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
    
    def log(self, msg: str, level: str = 'info'):
        """Log message only on rank 0."""
        if self.rank == 0:
            getattr(self.logger, level)(msg)
    
    def create_model(self) -> nn.Module:
        """Create and wrap model with DDP."""
        # Create model (example: ResNet)
        import torchvision.models as models
        
        if self.config.model_name == "resnet50":
            model = models.resnet50(num_classes=self.config.num_classes)
        elif self.config.model_name == "resnet101":
            model = models.resnet101(num_classes=self.config.num_classes)
        else:
            raise ValueError(f"Unknown model: {self.config.model_name}")
        
        # Move to device
        model = model.to(self.device)
        
        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
            # static_graph=True,  # Enable for PyTorch 2.0+ with fixed graph
        )
        
        return model
    
    def create_dataloader(self, train: bool = True) -> DataLoader:
        """Create distributed data loader."""
        import torchvision.transforms as transforms
        from torchvision.datasets import FakeData
        
        # Example transforms
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        # Use FakeData for demonstration (replace with your dataset)
        dataset = FakeData(
            size=10000 if train else 1000,
            image_size=(3, 224, 224),
            num_classes=self.config.num_classes,
            transform=transform
        )
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=train
        )
        
        # Create data loader
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=train
        )
        
        return loader, sampler
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer with learning rate scaling."""
        # Scale learning rate by world size (linear scaling rule)
        scaled_lr = self.config.lr * self.world_size
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=scaled_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        # Cosine annealing with warmup
        warmup_epochs = 5
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.5 * (1 + torch.cos(torch.tensor(
                (epoch - warmup_epochs) / (self.config.epochs - warmup_epochs) * 3.14159
            ))).item()
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: optim.lr_scheduler._LRScheduler,
                       epoch: int, best_acc: float, scaler: Optional[GradScaler] = None,
                       is_best: bool = False):
        """Save training checkpoint (only on rank 0)."""
        if self.rank != 0:
            return
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'config': vars(self.config),
        }
        
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        self.log(f"Saved checkpoint to {path}")
        
        # Save best model separately for deployment/inference
        if is_best:
            best_path = checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            self.log(f"Saved best model (acc={best_acc:.2f}%) to {best_path}")
    
    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: optim.lr_scheduler._LRScheduler,
                       scaler: Optional[GradScaler] = None) -> tuple:
        """Load checkpoint."""
        if self.config.resume_from is None:
            return 0, 0.0
        
        checkpoint_path = Path(self.config.resume_from)
        if not checkpoint_path.exists():
            self.log(f"Checkpoint not found: {checkpoint_path}", 'warning')
            return 0, 0.0
        
        # Load on CPU first, then move to device
        map_location = {'cuda:0': f'cuda:{self.local_rank}'}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        
        self.log(f"Resumed from epoch {start_epoch}, best_acc: {best_acc:.2f}%")
        
        return start_epoch, best_acc
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   scaler: Optional[GradScaler], epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        optimizer.zero_grad()
        
        for step, (inputs, targets) in enumerate(train_loader):
            if HAS_NVTX:
                nvtx.range_push(f"step_{step}")
            
            # Move to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if HAS_NVTX:
                nvtx.range_push("forward")
            
            if self.config.use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / self.config.accumulation_steps
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / self.config.accumulation_steps
            
            if HAS_NVTX:
                nvtx.range_pop()  # forward
            
            # Backward pass
            if HAS_NVTX:
                nvtx.range_push("backward")
            
            # Use no_sync for gradient accumulation (except last step)
            is_accumulation_step = (step + 1) % self.config.accumulation_steps != 0
            
            context = model.no_sync if is_accumulation_step else nullcontext
            
            with context():
                if self.config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            if HAS_NVTX:
                nvtx.range_pop()  # backward
            
            # Optimizer step (after accumulation)
            if not is_accumulation_step:
                if HAS_NVTX:
                    nvtx.range_push("optimizer")
                
                if self.config.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                if HAS_NVTX:
                    nvtx.range_pop()  # optimizer
            
            # Metrics
            total_loss += loss.item() * self.config.accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Logging
            if step % self.config.log_every == 0:
                self.log(f"Epoch {epoch}, Step {step}/{len(train_loader)}, "
                        f"Loss: {loss.item() * self.config.accumulation_steps:.4f}")
            
            if HAS_NVTX:
                nvtx.range_pop()  # step
        
        # Aggregate metrics across all processes
        metrics = self._aggregate_metrics(total_loss, correct, total, len(train_loader))
        
        return metrics
    
    def evaluate(self, model: nn.Module, val_loader: DataLoader,
                criterion: nn.Module) -> Dict[str, float]:
        """Evaluate model on validation set."""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        metrics = self._aggregate_metrics(total_loss, correct, total, len(val_loader))
        
        return metrics
    
    def _aggregate_metrics(self, loss: float, correct: int, total: int,
                          num_batches: int) -> Dict[str, float]:
        """Aggregate metrics across all processes."""
        # Create tensors for reduction
        metrics_tensor = torch.tensor(
            [loss, correct, total, num_batches],
            dtype=torch.float64,
            device=self.device
        )
        
        # All-reduce
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        loss, correct, total, num_batches = metrics_tensor.tolist()
        
        return {
            'loss': loss / num_batches,
            'accuracy': 100.0 * correct / total,
            'samples': int(total)
        }
    
    def train(self):
        """Main training loop."""
        # Create components
        model = self.create_model()
        train_loader, train_sampler = self.create_dataloader(train=True)
        val_loader, _ = self.create_dataloader(train=False)
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        scaler = GradScaler() if self.config.use_amp else None
        
        # Resume from checkpoint
        start_epoch, best_acc = self.load_checkpoint(model, optimizer, scheduler, scaler)
        
        self.log(f"Starting training from epoch {start_epoch}")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(start_epoch, self.config.epochs):
            epoch_start = time.time()
            
            # Set epoch for sampler
            train_sampler.set_epoch(epoch)
            
            # Train
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, criterion, scaler, epoch
            )
            
            # Evaluate
            val_metrics = self.evaluate(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.log(f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Train Acc={train_metrics['accuracy']:.2f}%, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val Acc={val_metrics['accuracy']:.2f}%, "
                    f"LR={scheduler.get_last_lr()[0]:.6f}, "
                    f"Time={epoch_time:.1f}s")
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > best_acc
            if is_best:
                best_acc = val_metrics['accuracy']
            
            if (epoch + 1) % self.config.save_every == 0 or is_best:
                self.save_checkpoint(model, optimizer, scheduler, epoch, best_acc, scaler, is_best)
        
        total_time = time.time() - start_time
        self.log(f"Training completed in {total_time / 3600:.2f} hours")
        self.log(f"Best validation accuracy: {best_acc:.2f}%")
        
        # Cleanup
        dist.destroy_process_group()


# Context manager for gradient accumulation
from contextlib import nullcontext


def main():
    parser = argparse.ArgumentParser(description='DDP Training')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        accumulation_steps=args.accumulation_steps,
        use_amp=not args.no_amp,
        resume_from=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    trainer = DistributedTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
