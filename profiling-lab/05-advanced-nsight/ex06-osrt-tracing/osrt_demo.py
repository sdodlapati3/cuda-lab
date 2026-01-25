#!/usr/bin/env python3
"""
Demo script with OS runtime behaviors worth tracing.
Shows file I/O, memory allocation, and threading patterns.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

# Optional NVTX for marking
try:
    import torch.cuda.nvtx as nvtx
except ImportError:
    class nvtx:
        @staticmethod
        def range_push(msg): pass
        @staticmethod
        def range_pop(): pass


def heavy_file_io(num_files=10, size_mb=10):
    """Simulate heavy file I/O operations."""
    nvtx.range_push("heavy_file_io")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write phase
        nvtx.range_push("write_files")
        for i in range(num_files):
            data = np.random.randn(size_mb * 1024 * 256).astype(np.float32)
            path = os.path.join(tmpdir, f"data_{i}.npy")
            np.save(path, data)
        nvtx.range_pop()
        
        # Read phase
        nvtx.range_push("read_files")
        for i in range(num_files):
            path = os.path.join(tmpdir, f"data_{i}.npy")
            data = np.load(path)
        nvtx.range_pop()
    
    nvtx.range_pop()


def memory_allocation_patterns():
    """Various memory allocation patterns."""
    nvtx.range_push("memory_patterns")
    
    # Many small allocations
    nvtx.range_push("small_allocs")
    tensors = []
    for _ in range(1000):
        tensors.append(torch.randn(100, 100, device='cuda'))
    nvtx.range_pop()
    
    # Few large allocations
    nvtx.range_push("large_allocs")
    large = torch.randn(5000, 5000, device='cuda')
    nvtx.range_pop()
    
    # Cleanup
    del tensors
    del large
    torch.cuda.empty_cache()
    
    nvtx.range_pop()


def threaded_preprocessing(num_threads=4):
    """Simulate threaded CPU preprocessing."""
    nvtx.range_push("threaded_preprocessing")
    
    def preprocess_batch(batch_id):
        """CPU-bound preprocessing."""
        data = np.random.randn(256, 224, 224, 3).astype(np.float32)
        # Simulate augmentation
        data = np.flip(data, axis=2)  # Horizontal flip
        data = (data - data.mean()) / (data.std() + 1e-7)  # Normalize
        return torch.from_numpy(data)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(preprocess_batch, i) for i in range(8)]
        results = [f.result() for f in futures]
    
    nvtx.range_pop()
    return results


def gpu_computation(data):
    """Run GPU computation on preprocessed data."""
    nvtx.range_push("gpu_computation")
    
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    ).cuda()
    
    for batch in data:
        batch = batch.cuda().permute(0, 3, 1, 2)  # NHWC -> NCHW
        with torch.no_grad():
            output = model(batch)
    
    nvtx.range_pop()


def main():
    print("OS Runtime Tracing Demo")
    print("=" * 50)
    
    print("\n1. Heavy File I/O...")
    heavy_file_io(num_files=5, size_mb=5)
    
    print("2. Memory Allocation Patterns...")
    memory_allocation_patterns()
    
    print("3. Threaded Preprocessing...")
    data = threaded_preprocessing(num_threads=4)
    
    print("4. GPU Computation...")
    gpu_computation(data)
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nKey things to observe in the profile:")
    print("- File read/write system calls in OS Runtime lane")
    print("- Memory allocation patterns (mmap, brk)")
    print("- Thread creation and synchronization")
    print("- Correlation between CPU threads and GPU activity")


if __name__ == "__main__":
    main()
