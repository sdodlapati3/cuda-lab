"""
Training script with CPU bottleneck for profiling exercise.

The bottleneck is in inefficient_preprocessing() - find it with CPU sampling!

Profile with:
    nsys profile --sample=cpu --trace=cuda -o cpu_bottleneck python training_with_cpu_bottleneck.py
"""

import torch
import torch.nn as nn
import argparse


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def inefficient_preprocessing(data: torch.Tensor) -> torch.Tensor:
    """
    CPU BOTTLENECK - This function wastes CPU time!
    
    Issues:
    1. Runs on CPU while GPU is idle
    2. Unnecessary loop iterations
    3. Non-vectorized operations
    
    Use CPU sampling to find this hotspot.
    """
    result = data.clone()
    
    # Wasteful CPU computation
    for i in range(50):
        result = result + 0.001
        result = torch.sin(result)
        result = torch.cos(result)
        # Each iteration is cheap, but 50 iterations adds up
    
    return result


def efficient_preprocessing(data: torch.Tensor) -> torch.Tensor:
    """
    Fixed version - no unnecessary iterations.
    """
    # Just normalize - quick and simple
    return (data - data.mean()) / (data.std() + 1e-6)


def create_dataloader(batch_size: int, input_dim: int, num_batches: int):
    """Generate synthetic data."""
    for _ in range(num_batches):
        data = torch.randn(batch_size, input_dim)
        target = torch.randint(0, 10, (batch_size,))
        yield data, target


def train_slow(model, dataloader, optimizer, criterion, device):
    """Training with CPU bottleneck."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # CPU BOTTLENECK HERE
        data = inefficient_preprocessing(data)
        
        # Move to GPU (GPU was idle during preprocessing!)
        data = data.to(device)
        target = target.to(device)
        
        # Standard training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss


def train_fast(model, dataloader, optimizer, criterion, device):
    """Training without CPU bottleneck."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # Efficient preprocessing
        data = efficient_preprocessing(data)
        
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['slow', 'fast'], default='slow')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-batches', type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_fn = train_slow if args.mode == 'slow' else train_fast
    
    import time
    for epoch in range(args.epochs):
        dataloader = create_dataloader(
            args.batch_size, 1024, args.num_batches
        )
        
        start = time.perf_counter()
        loss = train_fn(model, dataloader, optimizer, criterion, device)
        elapsed = time.perf_counter() - start
        
        print(f"Epoch {epoch+1}: Loss={loss/args.num_batches:.4f}, Time={elapsed:.2f}s")
    
    print("\nProfile this with:")
    print("  nsys profile --sample=cpu --trace=cuda -o cpu_profile python training_with_cpu_bottleneck.py")
    print("\nThen look for inefficient_preprocessing in CPU sampling results!")


if __name__ == "__main__":
    main()
