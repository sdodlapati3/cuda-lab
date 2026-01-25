"""
Training script with intentional performance issue for profiling exercise.

Profile this script to find the bottleneck using Python backtraces:

    nsys profile --python-backtrace=cuda -o profile python train_model.py
"""

import torch
import torch.nn as nn
import argparse
from model import TransformerWithHeavyLayer


def create_dummy_dataloader(batch_size: int, seq_len: int, vocab_size: int = 10000, num_batches: int = 20):
    """Create dummy data for profiling."""
    for _ in range(num_batches):
        data = torch.randint(0, vocab_size, (batch_size, seq_len))
        target = torch.randint(0, vocab_size, (batch_size,))
        yield data, target


def training_step(model, data, target, optimizer, criterion):
    """Single training step - the profiler will trace into this."""
    optimizer.zero_grad()
    
    # Forward pass - which layer is slow?
    output = model(data)
    
    # Loss computation
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()


def train_epoch(model, dataloader, optimizer, criterion, device, max_batches=None):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        target = target.to(device)
        
        loss = training_step(model, data, target, optimizer, criterion)
        total_loss += loss
        
        if max_batches and batch_idx >= max_batches:
            break
    
    return total_loss / (batch_idx + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--max-batches', type=int, default=15)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with intentional performance issue
    model = TransformerWithHeavyLayer(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        dataloader = create_dummy_dataloader(
            args.batch_size, 
            args.seq_len,
            num_batches=args.max_batches + 5
        )
        
        avg_loss = train_epoch(
            model, dataloader, optimizer, criterion, 
            device, max_batches=args.max_batches
        )
        
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}")
    
    print("Training complete. Check profiler output for bottlenecks.")


if __name__ == "__main__":
    main()
