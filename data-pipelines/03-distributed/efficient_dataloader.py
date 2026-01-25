#!/usr/bin/env python3
"""
efficient_dataloader.py - Optimized data loading for distributed training

Demonstrates:
- DataLoader configuration for maximum throughput
- Prefetching and pinned memory
- Non-blocking GPU transfers
- Profiling data loading performance

Usage:
    torchrun --nproc_per_node=4 efficient_dataloader.py
    torchrun --nproc_per_node=4 efficient_dataloader.py --profile

Author: CUDA Lab
"""

import os
import time
import argparse
from typing import Dict, Iterator, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler


@dataclass
class DataLoaderConfig:
    """Configuration for optimized DataLoader."""
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = True
    non_blocking: bool = True


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


# ============================================================================
# Datasets
# ============================================================================

class SyntheticImageDataset(Dataset):
    """Synthetic image dataset simulating real data loading."""
    
    def __init__(self, size: int = 10000, image_size: int = 224, simulate_io: bool = True):
        self.size = size
        self.image_size = image_size
        self.simulate_io = simulate_io
        
        # Pre-generate some data
        self.labels = torch.randint(0, 1000, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate I/O delay
        if self.simulate_io:
            time.sleep(0.001)  # 1ms simulated I/O
        
        # Generate image (in real case, load from disk)
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Simulate preprocessing
        image = image.clamp(-3, 3) / 3  # Normalize
        
        return {
            "image": image,
            "label": self.labels[idx],
            "idx": idx,
        }


class FastSyntheticDataset(Dataset):
    """Pre-allocated dataset for maximum throughput testing."""
    
    def __init__(self, size: int = 10000, image_size: int = 224):
        self.size = size
        # Pre-allocate all data in memory
        self.images = torch.randn(size, 3, image_size, image_size)
        self.labels = torch.randint(0, 1000, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
        }


# ============================================================================
# Optimized DataLoader Creation
# ============================================================================

def create_optimized_dataloader(
    dataset: Dataset,
    config: DataLoaderConfig,
    rank: int,
    world_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    """Create an optimized distributed DataLoader."""
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=config.drop_last,
    )
    
    # Determine if we can use persistent workers
    # (requires num_workers > 0)
    use_persistent = config.persistent_workers and config.num_workers > 0
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=use_persistent,
        drop_last=config.drop_last,
    )
    
    return loader, sampler


# ============================================================================
# GPU Data Prefetcher
# ============================================================================

class CUDAPrefetcher:
    """
    Prefetches batches to GPU using CUDA streams.
    
    Overlaps data transfer with compute for better utilization.
    """
    
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        non_blocking: bool = True,
    ):
        self.loader = loader
        self.device = device
        self.non_blocking = non_blocking
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self.iterator = None
    
    def __iter__(self):
        self.iterator = iter(self.loader)
        self._preload()
        return self
    
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        
        # Record that batch is being used
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                v.record_stream(torch.cuda.current_stream())
        
        self._preload()
        return batch
    
    def _preload(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_batch = {
                k: v.to(self.device, non_blocking=self.non_blocking)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
    
    def __len__(self):
        return len(self.loader)


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_dataloader(
    name: str,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 100,
    use_prefetcher: bool = False,
    warmup_batches: int = 10,
) -> Dict:
    """Benchmark DataLoader throughput."""
    
    # Create iterator
    if use_prefetcher:
        data_iter = CUDAPrefetcher(loader, device)
    else:
        data_iter = iter(loader)
    
    # Warmup
    for i, batch in enumerate(data_iter):
        if not use_prefetcher:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        if i >= warmup_batches:
            break
    
    # Reset iterator
    if use_prefetcher:
        data_iter = CUDAPrefetcher(loader, device)
    else:
        data_iter = iter(loader)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    samples = 0
    batches = 0
    
    for i, batch in enumerate(data_iter):
        if not use_prefetcher:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        # Simulate a small compute operation
        if "image" in batch:
            _ = batch["image"].mean()
        
        samples += batch["label"].size(0) if "label" in batch else len(batch.get("idx", []))
        batches += 1
        
        if batches >= num_batches:
            break
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return {
        "name": name,
        "samples": samples,
        "batches": batches,
        "time": elapsed,
        "samples_per_sec": samples / elapsed,
        "batches_per_sec": batches / elapsed,
    }


def compare_configurations(rank: int, world_size: int, device: torch.device):
    """Compare different DataLoader configurations."""
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("DATALOADER CONFIGURATION COMPARISON")
        print("=" * 70)
    
    dataset = SyntheticImageDataset(size=5000, simulate_io=True)
    
    configs = [
        ("Baseline (1 worker)", DataLoaderConfig(num_workers=1, pin_memory=False)),
        ("4 workers", DataLoaderConfig(num_workers=4, pin_memory=False)),
        ("4 workers + pin_memory", DataLoaderConfig(num_workers=4, pin_memory=True)),
        ("4 workers + pin + prefetch", DataLoaderConfig(num_workers=4, pin_memory=True, prefetch_factor=4)),
        ("8 workers + pin + prefetch", DataLoaderConfig(num_workers=8, pin_memory=True, prefetch_factor=4)),
    ]
    
    results = []
    
    for name, config in configs:
        loader, sampler = create_optimized_dataloader(
            dataset, config, rank, world_size
        )
        sampler.set_epoch(0)
        
        result = benchmark_dataloader(name, loader, device, num_batches=50)
        results.append(result)
        
        if rank == 0:
            print(f"{name:<35}: {result['samples_per_sec']:>8.1f} samples/s")
    
    # Test with prefetcher
    if rank == 0:
        print("\nWith CUDA Prefetcher:")
    
    config = DataLoaderConfig(num_workers=4, pin_memory=True, prefetch_factor=4)
    loader, sampler = create_optimized_dataloader(dataset, config, rank, world_size)
    sampler.set_epoch(0)
    
    result = benchmark_dataloader(
        "4 workers + CUDA prefetch",
        loader, device,
        num_batches=50,
        use_prefetcher=True
    )
    
    if rank == 0:
        print(f"{'4 workers + CUDA prefetch':<35}: {result['samples_per_sec']:>8.1f} samples/s")
    
    return results


def profile_dataloader(rank: int, loader: DataLoader, device: torch.device):
    """Profile DataLoader with PyTorch profiler."""
    if rank != 0:
        return
    
    print("\n" + "=" * 70)
    print("DATALOADER PROFILING")
    print("=" * 70)
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            # Simulate compute
            _ = batch["image"].mean()
            torch.cuda.synchronize()
            
            if i >= 20:
                break
    
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=15,
    ))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Efficient DataLoader Demo")
    parser.add_argument("--profile", action="store_true", help="Run profiler")
    parser.add_argument("--num-batches", type=int, default=50)
    args = parser.parse_args()
    
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print("=" * 70)
        print("EFFICIENT DISTRIBUTED DATA LOADING")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name()}")
    
    # Compare configurations
    compare_configurations(rank, world_size, device)
    
    # Profiling
    if args.profile:
        config = DataLoaderConfig(num_workers=4, pin_memory=True)
        dataset = SyntheticImageDataset(size=5000, simulate_io=True)
        loader, sampler = create_optimized_dataloader(
            dataset, config, rank, world_size
        )
        sampler.set_epoch(0)
        profile_dataloader(rank, loader, device)
    
    # Print recommendations
    if rank == 0:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        print("""
1. num_workers:
   - Start with 4 per GPU
   - Increase if CPU utilization is low
   - Watch RAM usage (workers * prefetch * batch * sample_size)

2. pin_memory=True:
   - Always use for GPU training
   - Enables async CPU→GPU transfer

3. prefetch_factor:
   - Default is 2
   - Increase to 4-8 if I/O is slow
   - Higher = more RAM usage

4. persistent_workers=True:
   - Avoids worker restart between epochs
   - Saves ~1-2s per epoch

5. CUDA Prefetcher:
   - Use for maximum throughput
   - Overlaps transfer with compute

6. Common issues:
   - Low GPU utilization → increase num_workers
   - High RAM usage → decrease prefetch_factor
   - Inconsistent batches → use drop_last=True
""")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
