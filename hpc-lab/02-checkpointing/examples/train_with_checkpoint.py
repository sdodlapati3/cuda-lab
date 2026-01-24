"""
train_with_checkpoint.py - Fault-tolerant training example

Demonstrates:
- Automatic checkpointing every N steps
- Resume from latest checkpoint
- Signal handling for preemption
- Distributed training support

Usage:
    # Single GPU
    python train_with_checkpoint.py --checkpoint-dir ./checkpoints

    # Resume training
    python train_with_checkpoint.py --checkpoint-dir ./checkpoints --resume

    # Multi-GPU
    torchrun --nproc_per_node=4 train_with_checkpoint.py --checkpoint-dir ./checkpoints

    # With Slurm (auto-resume on preemption)
    sbatch auto_resume.sbatch

Author: CUDA Lab
"""

import os
import argparse
import signal
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# Import our checkpoint utilities
from checkpoint_utils import CheckpointManager, is_main_process


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""
    
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual
        x = self.norm2(x + self.ff(x))
        return x


class DemoModel(nn.Module):
    """Demo model with transformer blocks."""
    
    def __init__(self, vocab_size=10000, d_model=512, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        # Single GPU fallback
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_dummy_data(num_samples=10000, seq_len=128, vocab_size=10000):
    """Create dummy dataset."""
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TensorDataset(X, y)


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch,
    checkpoint_manager,
    checkpoint_every=100,
    start_step=0,
):
    """
    Train for one epoch with checkpointing.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        epoch: Current epoch number
        checkpoint_manager: CheckpointManager instance
        checkpoint_every: Steps between checkpoints
        start_step: Step to resume from (for partial epoch resume)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for step, (data, target) in enumerate(train_loader):
        # Skip steps if resuming mid-epoch
        if step < start_step:
            continue
        
        # Check for preemption signal
        if checkpoint_manager.should_save_on_signal():
            if is_main_process():
                print(f"Preemption signal received at step {step}")
            checkpoint_manager.save(
                model, optimizer, scheduler, epoch,
                step=step, loss=total_loss / max(num_batches, 1),
                extra_data={'preempted': True}
            )
            return total_loss / max(num_batches, 1)
        
        # Move data to device
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        
        # Compute loss
        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Periodic checkpoint
        if (step + 1) % checkpoint_every == 0:
            avg_loss = total_loss / num_batches
            
            # Save checkpoint
            checkpoint_manager.save(
                model, optimizer, scheduler, epoch,
                step=step + 1, loss=avg_loss
            )
            
            if is_main_process():
                print(f"Epoch {epoch}, Step {step+1}, Loss: {avg_loss:.4f}")
        
        # Log progress
        if step % 10 == 0 and is_main_process():
            print(f"  Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
    
    # End of epoch - step scheduler and checkpoint
    if scheduler is not None:
        scheduler.step()
    
    avg_loss = total_loss / max(num_batches, 1)
    checkpoint_manager.save(model, optimizer, scheduler, epoch + 1, loss=avg_loss)
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Fault-tolerant training demo')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--max-checkpoints', type=int, default=3)
    args = parser.parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    if is_main_process():
        print(f"Training with {world_size} GPU(s)")
        print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Create model
    model = DemoModel().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        max_checkpoints=args.max_checkpoints,
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    start_step = 0
    if args.resume:
        checkpoint_data = checkpoint_manager.load(model, optimizer, scheduler)
        if checkpoint_data is not None:
            start_epoch = checkpoint_data.get('epoch', 0)
            start_step = checkpoint_data.get('step', 0)
            if is_main_process():
                print(f"Resumed from epoch {start_epoch}, step {start_step}")
    
    # Create dataset and dataloader
    dataset = create_dummy_data()
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            
            if is_main_process():
                print(f"\n=== Epoch {epoch} ===")
            
            # Determine start step for this epoch
            epoch_start_step = start_step if epoch == start_epoch else 0
            
            avg_loss = train_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                device, epoch, checkpoint_manager,
                checkpoint_every=args.checkpoint_every,
                start_step=epoch_start_step,
            )
            
            if is_main_process():
                print(f"Epoch {epoch} complete. Average loss: {avg_loss:.4f}")
        
        if is_main_process():
            print("\n=== Training Complete ===")
            print(f"Final model saved to {args.checkpoint_dir}")
    
    except Exception as e:
        # Save emergency checkpoint on error
        if is_main_process():
            print(f"Error during training: {e}")
            checkpoint_manager.save(
                model, optimizer, scheduler, epoch,
                extra_data={'error': str(e)}
            )
        raise
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
