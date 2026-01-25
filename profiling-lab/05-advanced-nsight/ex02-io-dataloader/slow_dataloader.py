"""
Slow DataLoader example with common performance issues.

This demonstrates typical data loading bottlenecks:
1. No caching - loads from disk every time
2. Heavy CPU preprocessing in __getitem__
3. Single worker (no parallelism)
4. No pinned memory
"""

import torch
from torch.utils.data import Dataset, DataLoader
import time
import os
from pathlib import Path


class SlowDataset(Dataset):
    """
    Dataset with intentional performance issues.
    
    Issues:
    - Loads from disk on every access (no caching)
    - Heavy preprocessing in __getitem__ (blocks training)
    - Creates unnecessary tensor copies
    """
    
    def __init__(self, num_samples: int = 1000, data_dim: int = 1024):
        self.num_samples = num_samples
        self.data_dim = data_dim
        
        # Create temp data files (simulating real dataset)
        self.data_dir = Path("/tmp/slow_dataset")
        self.data_dir.mkdir(exist_ok=True)
        
        print("Creating dataset files...")
        for i in range(num_samples):
            data = torch.randn(data_dim)
            label = torch.randint(0, 10, (1,)).item()
            torch.save({'data': data, 'label': label}, self.data_dir / f"sample_{i}.pt")
        print(f"Created {num_samples} files")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Issue 1: Load from disk every time (no caching!)
        sample = torch.load(self.data_dir / f"sample_{idx}.pt")
        data = sample['data']
        label = sample['label']
        
        # Issue 2: Heavy CPU preprocessing (simulated)
        data = self._heavy_preprocess(data)
        
        # Issue 3: Unnecessary tensor copy
        data = torch.tensor(data.numpy())  # Don't do this!
        
        return data, label
    
    def _heavy_preprocess(self, data):
        """Simulated heavy preprocessing - wastes CPU time."""
        # Simulate CPU-intensive work
        for _ in range(100):
            data = torch.fft.fft(data.to(torch.complex64)).real
            data = torch.nn.functional.normalize(data, dim=0)
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


def train_slow(num_epochs: int = 2, batch_size: int = 32, num_samples: int = 200):
    """Training with slow data loading."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataset and slow dataloader
    dataset = SlowDataset(num_samples=num_samples)
    
    # Issue 4: num_workers=0 means no parallel loading
    # Issue 5: pin_memory=False means slower H2D transfers
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # NO PARALLELISM!
        pin_memory=False    # SLOW TRANSFERS!
    )
    
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\nTraining with SLOW DataLoader:")
    print(f"  num_workers=0, pin_memory=False")
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
            
            # To device
            data = data.to(device)
            labels = torch.tensor(labels).to(device)
            
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
    
    # Cleanup
    import shutil
    shutil.rmtree(dataset.data_dir, ignore_errors=True)


if __name__ == "__main__":
    train_slow()
