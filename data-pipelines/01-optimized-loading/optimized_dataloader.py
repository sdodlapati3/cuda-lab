"""
optimized_dataloader.py - High-Performance PyTorch DataLoader Patterns

Demonstrates various optimizations for data loading:
1. Worker configuration
2. Memory pinning
3. Prefetching
4. Persistent workers
5. Custom collate functions

Author: CUDA Lab
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import time
import argparse
from typing import Optional, Tuple, List, Callable
import os
from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    """Configuration for optimized DataLoader."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = True


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for benchmarking data loading.
    Simulates various data loading patterns.
    """
    
    def __init__(
        self,
        size: int,
        input_shape: Tuple[int, ...],
        simulate_io_ms: float = 0.0,
        simulate_cpu_ms: float = 0.0
    ):
        """
        Args:
            size: Number of samples
            input_shape: Shape of each sample (e.g., (3, 224, 224))
            simulate_io_ms: Simulated I/O latency per sample
            simulate_cpu_ms: Simulated CPU preprocessing time
        """
        self.size = size
        self.input_shape = input_shape
        self.simulate_io_ms = simulate_io_ms
        self.simulate_cpu_ms = simulate_cpu_ms
        
        # Pre-generate data for fair comparison
        self.data = torch.randn(size, *input_shape)
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate I/O latency
        if self.simulate_io_ms > 0:
            time.sleep(self.simulate_io_ms / 1000)
        
        x = self.data[idx]
        
        # Simulate CPU preprocessing
        if self.simulate_cpu_ms > 0:
            time.sleep(self.simulate_cpu_ms / 1000)
            # Simulate some actual work
            x = x * 2.0 - 1.0  # Normalize
        
        return x, self.labels[idx]


class MemoryMappedDataset(Dataset):
    """
    Memory-mapped dataset for efficient random access.
    Uses numpy memmap for data that doesn't fit in RAM.
    """
    
    def __init__(
        self,
        data_path: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32
    ):
        self.data = np.memmap(data_path, dtype=dtype, mode='r', shape=shape)
        self.n_samples = shape[0]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Memory-mapped read is lazy - only loads requested data
        x = torch.from_numpy(self.data[idx].copy())
        return x


class ChunkedIterableDataset(IterableDataset):
    """
    Iterable dataset that reads data in chunks.
    Good for streaming large datasets.
    """
    
    def __init__(
        self,
        n_samples: int,
        input_shape: Tuple[int, ...],
        chunk_size: int = 1000
    ):
        self.n_samples = n_samples
        self.input_shape = input_shape
        self.chunk_size = chunk_size
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process
            start, end = 0, self.n_samples
        else:
            # Multi-process: split work
            per_worker = self.n_samples // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker
        
        # Stream chunks
        for chunk_start in range(start, end, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, end)
            
            # Generate or load chunk
            chunk_data = torch.randn(chunk_end - chunk_start, *self.input_shape)
            chunk_labels = torch.randint(0, 10, (chunk_end - chunk_start,))
            
            for i in range(len(chunk_data)):
                yield chunk_data[i], chunk_labels[i]


def create_optimized_dataloader(
    dataset: Dataset,
    config: DataLoaderConfig,
    shuffle: bool = True
) -> DataLoader:
    """Create an optimized DataLoader with best practices."""
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=config.drop_last,
    )


def fast_collate_fn(batch):
    """
    Custom collate function optimized for speed.
    Avoids unnecessary copies.
    """
    # Separate inputs and labels
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Stack efficiently
    inputs = torch.stack(inputs)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return inputs, labels


def benchmark_dataloader(
    dataloader: DataLoader,
    n_batches: int = 100,
    device: torch.device = None,
    warmup_batches: int = 10
) -> dict:
    """
    Benchmark DataLoader throughput.
    
    Returns:
        dict with timing statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_times = []
    transfer_times = []
    samples_seen = 0
    
    data_iter = iter(dataloader)
    
    # Warmup
    for _ in range(warmup_batches):
        try:
            batch = next(data_iter)
            if device.type == 'cuda':
                batch[0].to(device)
        except StopIteration:
            data_iter = iter(dataloader)
    
    # Benchmark
    start_total = time.perf_counter()
    
    for i in range(n_batches):
        try:
            start_batch = time.perf_counter()
            inputs, labels = next(data_iter)
            load_time = time.perf_counter() - start_batch
            
            start_transfer = time.perf_counter()
            if device.type == 'cuda':
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                torch.cuda.synchronize()
            transfer_time = time.perf_counter() - start_transfer
            
            batch_times.append(load_time)
            transfer_times.append(transfer_time)
            samples_seen += inputs.shape[0]
            
        except StopIteration:
            data_iter = iter(dataloader)
    
    total_time = time.perf_counter() - start_total
    
    return {
        'total_time_s': total_time,
        'samples_per_sec': samples_seen / total_time,
        'batches_per_sec': n_batches / total_time,
        'mean_batch_time_ms': np.mean(batch_times) * 1000,
        'std_batch_time_ms': np.std(batch_times) * 1000,
        'mean_transfer_time_ms': np.mean(transfer_times) * 1000,
        'total_samples': samples_seen,
    }


def find_optimal_workers(
    dataset: Dataset,
    batch_size: int = 32,
    max_workers: int = 16,
    device: torch.device = None
) -> int:
    """Find optimal number of workers by benchmarking."""
    
    print("\nFinding optimal number of workers...")
    print("-" * 50)
    
    best_throughput = 0
    best_workers = 0
    
    for num_workers in range(0, max_workers + 1, 2):
        if num_workers == 0:
            num_workers = 0  # Test with 0 workers
        
        config = DataLoaderConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        )
        
        dataloader = create_optimized_dataloader(dataset, config)
        
        try:
            results = benchmark_dataloader(
                dataloader, 
                n_batches=50, 
                device=device,
                warmup_batches=5
            )
            
            throughput = results['samples_per_sec']
            print(f"  Workers={num_workers:2d}: {throughput:8.1f} samples/sec")
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_workers = num_workers
                
        except Exception as e:
            print(f"  Workers={num_workers:2d}: ERROR - {e}")
        
        # Clean up
        del dataloader
    
    print(f"\nOptimal: {best_workers} workers ({best_throughput:.1f} samples/sec)")
    return best_workers


def compare_configurations(device: torch.device):
    """Compare different DataLoader configurations."""
    
    print("\n" + "="*60)
    print("DataLoader Configuration Comparison")
    print("="*60)
    
    # Create dataset with simulated I/O
    dataset = SyntheticDataset(
        size=10000,
        input_shape=(3, 224, 224),
        simulate_io_ms=1.0,  # 1ms simulated I/O
        simulate_cpu_ms=0.5   # 0.5ms simulated preprocessing
    )
    
    configs = {
        'Baseline (0 workers)': DataLoaderConfig(
            batch_size=32,
            num_workers=0,
            pin_memory=False,
        ),
        'Pin Memory': DataLoaderConfig(
            batch_size=32,
            num_workers=0,
            pin_memory=True,
        ),
        '4 Workers': DataLoaderConfig(
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        ),
        '8 Workers': DataLoaderConfig(
            batch_size=32,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
        ),
        '8 Workers + Persistent': DataLoaderConfig(
            batch_size=32,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        ),
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\nTesting: {name}")
        
        dataloader = create_optimized_dataloader(dataset, config)
        
        try:
            result = benchmark_dataloader(
                dataloader, 
                n_batches=100,
                device=device
            )
            results[name] = result
            
            print(f"  Throughput: {result['samples_per_sec']:,.0f} samples/sec")
            print(f"  Batch time: {result['mean_batch_time_ms']:.2f} ± "
                  f"{result['std_batch_time_ms']:.2f} ms")
            
        except Exception as e:
            print(f"  ERROR: {e}")
        
        del dataloader
    
    # Print summary
    if results:
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        
        baseline = results.get('Baseline (0 workers)', {}).get('samples_per_sec', 1)
        
        for name, result in results.items():
            speedup = result['samples_per_sec'] / baseline
            print(f"  {name:30s}: {result['samples_per_sec']:8,.0f} samples/sec "
                  f"({speedup:.2f}x)")


def main():
    parser = argparse.ArgumentParser(description='Optimized DataLoader Benchmark')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run configuration comparison')
    parser.add_argument('--find-workers', action='store_true',
                       help='Find optimal worker count')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--input-shape', type=int, nargs='+', 
                       default=[3, 224, 224])
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if args.benchmark:
        compare_configurations(device)
    
    if args.find_workers:
        dataset = SyntheticDataset(
            size=10000,
            input_shape=tuple(args.input_shape),
            simulate_io_ms=1.0
        )
        find_optimal_workers(dataset, args.batch_size, device=device)
    
    if not args.benchmark and not args.find_workers:
        # Default demo
        compare_configurations(device)


if __name__ == "__main__":
    main()
