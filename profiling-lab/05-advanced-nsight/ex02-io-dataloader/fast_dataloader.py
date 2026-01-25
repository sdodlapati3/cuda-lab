"""
Fast DataLoader with optimizations.

Fixes:
1. Caches data in memory
2. Minimal preprocessing in __getitem__
3. Multiple workers for parallel loading
4. Pinned memory for faster transfers
5. Prefetching
"""

import torch
from torch.utils.data import Dataset, DataLoader
import time
from pathlib import Path


class FastDataset(Dataset):
    """
    Optimized dataset with proper caching.
    """
    
    def __init__(self, num_samples: int = 1000, data_dim: int = 1024, cache_in_memory: bool = True):
        self.num_samples = num_samples
        self.data_dim = data_dim
        
        # Pre-generate all data in memory (no disk I/O in __getitem__)
        print("Pre-generating dataset in memory...")
        self.data = torch.randn(num_samples, data_dim)
        self.labels = torch.randint(0, 10, (num_samples,))
        
        # Pre-process everything upfront if needed
        if cache_in_memory:
            self.data = self._preprocess_all(self.data)
        
        print(f"Dataset ready: {num_samples} samples in memory")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Fast: just index into pre-loaded tensors
        return self.data[idx], self.labels[idx]
    
    def _preprocess_all(self, data):
        """Batch preprocessing (much faster than per-sample)."""
        # Normalize all at once
        data = torch.nn.functional.normalize(data, dim=1)
        return data


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim: int = 1024, num_classes: int = 10):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def train_fast(num_epochs: int = 2, batch_size: int = 32, num_samples: int = 200, num_workers: int = 4):
    """Training with optimized data loading."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataset and fast dataloader
    dataset = FastDataset(num_samples=num_samples)
    
    # Optimized DataLoader settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,     # Parallel loading
        pin_memory=True,             # Faster H2D transfer
        prefetch_factor=2,           # Prefetch batches
        persistent_workers=True      # Keep workers alive between epochs
    )
    
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\nTraining with FAST DataLoader:")
    print(f"  num_workers={num_workers}, pin_memory=True, prefetch_factor=2")
    print()
    
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        data_time = 0.0
        train_time = 0.0
        
        data_start = time.perf_counter()
        for batch_idx, (data, labels) in enumerate(dataloader):
            data_time += time.perf_counter() - data_start
            
            train_start = time.perf_counter()
            
            # To device (faster with pinned memory)
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_time += time.perf_counter() - train_start
            
            data_start = time.perf_counter()
        
        epoch_time = time.perf_counter() - epoch_start
        
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Total time:  {epoch_time:.2f}s")
        print(f"  Data time:   {data_time:.2f}s ({100*data_time/epoch_time:.1f}%)")
        print(f"  Train time:  {train_time:.2f}s ({100*train_time/epoch_time:.1f}%)")
        print(f"  Loss: {total_loss/len(dataloader):.4f}")
        print()


if __name__ == "__main__":
    train_fast()
