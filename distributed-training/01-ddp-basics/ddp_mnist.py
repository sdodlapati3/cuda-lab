#!/usr/bin/env python3
"""
ddp_mnist.py - Simple DDP example with MNIST

A minimal but complete example of DistributedDataParallel training.

Usage:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 ddp_mnist.py
    
    # Multi-node (run on each node)
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
        --master_addr=10.0.0.1 --master_port=29500 ddp_mnist.py

Author: CUDA Lab
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def setup_distributed():
    """Initialize distributed training environment."""
    # torchrun sets these environment variables
    dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def get_data_loaders(batch_size: int, rank: int, world_size: int):
    """Create distributed data loaders for MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download only on rank 0
    if rank == 0:
        datasets.MNIST('./data', train=True, download=True)
        datasets.MNIST('./data', train=False, download=True)
    dist.barrier()  # Wait for rank 0 to download
    
    # Create datasets
    train_dataset = datasets.MNIST('./data', train=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_sampler


def train_epoch(model, train_loader, optimizer, epoch, local_rank, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to GPU
        data = data.to(local_rank, non_blocking=True)
        target = target.to(local_rank, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
        # Backward pass (gradients are automatically all-reduced)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log progress (only rank 0)
        if rank == 0 and batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset) // dist.get_world_size()}] '
                  f'Loss: {loss.item():.6f}')
    
    return total_loss / num_batches


def evaluate(model, test_loader, local_rank):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(local_rank, non_blocking=True)
            target = target.to(local_rank, non_blocking=True)
            
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    # Aggregate results across all processes
    test_loss_tensor = torch.tensor([test_loss], device=local_rank)
    correct_tensor = torch.tensor([correct], device=local_rank)
    total_tensor = torch.tensor([total], device=local_rank)
    
    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    test_loss = test_loss_tensor.item() / total_tensor.item()
    accuracy = 100. * correct_tensor.item() / total_tensor.item()
    
    return test_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='DDP MNIST Training')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size per GPU (default: 64)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save-model', action='store_true',
                        help='save the trained model')
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank, rank, world_size = setup_distributed()
    
    if rank == 0:
        print(f"Starting DDP training with {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
    
    # Create model and move to GPU
    model = SimpleCNN().to(local_rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    # Get data loaders
    train_loader, test_loader, train_sampler = get_data_loaders(
        args.batch_size, rank, world_size
    )
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        # Train
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, epoch, local_rank, rank)
        epoch_time = time.time() - epoch_start
        
        # Evaluate
        test_loss, accuracy = evaluate(model, test_loader, local_rank)
        
        if rank == 0:
            print(f'\nEpoch {epoch} Summary:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Test Loss: {test_loss:.4f}')
            print(f'  Accuracy: {accuracy:.2f}%')
            print(f'  Epoch Time: {epoch_time:.2f}s')
            print()
    
    total_time = time.time() - start_time
    
    if rank == 0:
        print(f"Training completed in {total_time:.2f}s")
        print(f"Final Accuracy: {accuracy:.2f}%")
        
        # Calculate throughput
        samples_per_epoch = len(train_loader.dataset)
        throughput = (samples_per_epoch * args.epochs) / total_time
        print(f"Throughput: {throughput:.1f} samples/sec")
        
        # Save model
        if args.save_model:
            torch.save(model.module.state_dict(), 'ddp_mnist_model.pt')
            print("Model saved to ddp_mnist_model.pt")
    
    # Cleanup
    cleanup()


if __name__ == '__main__':
    main()
