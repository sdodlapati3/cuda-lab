#!/usr/bin/env python3
"""
distributed_sampler.py - Distributed sampling patterns for multi-GPU training

Demonstrates:
- Standard DistributedSampler usage
- Custom samplers for imbalanced datasets
- Curriculum learning with distributed sampling
- Infinite samplers for streaming

Usage:
    torchrun --nproc_per_node=4 distributed_sampler.py

Author: CUDA Lab
"""

import os
import math
import random
from typing import Iterator, List, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


T_co = TypeVar('T_co', covariant=True)


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


# ============================================================================
# Sample Datasets
# ============================================================================

class SimpleDataset(Dataset):
    """Simple dataset for demonstration."""
    
    def __init__(self, size: int = 1000):
        self.size = size
        self.data = torch.arange(size)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {"idx": idx, "data": self.data[idx]}


class ImbalancedDataset(Dataset):
    """Dataset with class imbalance for weighted sampling demo."""
    
    def __init__(self, size: int = 1000, num_classes: int = 10):
        self.size = size
        self.num_classes = num_classes
        
        # Create imbalanced distribution (exponential decay)
        self.labels = []
        for i in range(size):
            # Class 0 is most common, class 9 is rarest
            class_id = int(random.random() ** 2 * num_classes)
            self.labels.append(class_id)
        
        self.labels = torch.tensor(self.labels)
        self.data = torch.randn(size, 10)
        
        # Compute class counts for weighted sampling
        self.class_counts = torch.bincount(self.labels, minlength=num_classes)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "label": self.labels[idx],
        }


# ============================================================================
# Custom Distributed Samplers
# ============================================================================

class DistributedWeightedSampler(Sampler[int]):
    """
    Distributed sampler with sample weights for class balancing.
    
    Each sample has a weight; samples with higher weights are more likely
    to be selected. Useful for handling imbalanced datasets.
    """
    
    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.weights = weights
        self.num_samples = num_samples
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0
        
        # Calculate samples per replica
        self.num_samples_per_replica = math.ceil(num_samples / num_replicas)
        self.total_size = self.num_samples_per_replica * num_replicas
    
    def __iter__(self) -> Iterator[int]:
        # Set seed for reproducibility across ranks
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Sample with weights
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=g,
        ).tolist()
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples_per_replica
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch


class DistributedCurriculumSampler(Sampler[int]):
    """
    Distributed curriculum sampler that presents easier examples first.
    
    Samples are sorted by difficulty and progressively includes
    harder samples as training progresses.
    """
    
    def __init__(
        self,
        difficulties: torch.Tensor,  # Lower = easier
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.difficulties = difficulties
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.progress = 0.0  # 0 to 1, how much of curriculum to use
        
        # Sort indices by difficulty
        self.sorted_indices = torch.argsort(difficulties).tolist()
        
        self.num_samples = len(difficulties)
        self.num_samples_per_replica = math.ceil(self.num_samples / num_replicas)
        self.total_size = self.num_samples_per_replica * num_replicas
    
    def set_progress(self, progress: float):
        """Set curriculum progress (0=easy only, 1=full dataset)."""
        self.progress = min(1.0, max(0.0, progress))
    
    def __iter__(self) -> Iterator[int]:
        # Determine how many samples to include based on progress
        # Start with 30% easiest, grow to 100%
        min_frac = 0.3
        frac = min_frac + (1 - min_frac) * self.progress
        num_active = int(len(self.sorted_indices) * frac)
        
        # Get active indices (easiest samples)
        active_indices = self.sorted_indices[:num_active]
        
        # Shuffle if needed
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(active_indices), generator=g)
            active_indices = [active_indices[i] for i in perm]
        
        # Pad to total size
        indices = active_indices.copy()
        while len(indices) < self.total_size:
            indices.extend(active_indices[:self.total_size - len(indices)])
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples_per_replica
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch


class InfiniteDistributedSampler(Sampler[int]):
    """
    Infinite sampler for streaming/continuous training.
    
    Yields indices infinitely with proper sharding across ranks.
    """
    
    def __init__(
        self,
        dataset_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
    
    def __iter__(self) -> Iterator[int]:
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            if self.shuffle:
                indices = torch.randperm(self.dataset_size, generator=g).tolist()
            else:
                indices = list(range(self.dataset_size))
            
            # Pad to be divisible by world_size
            padding = self.num_replicas - len(indices) % self.num_replicas
            if padding < self.num_replicas:
                indices += indices[:padding]
            
            # Yield this rank's portion
            for idx in indices[self.rank::self.num_replicas]:
                yield idx
            
            self.epoch += 1
    
    def __len__(self) -> int:
        # Return one epoch's worth
        return math.ceil(self.dataset_size / self.num_replicas)


# ============================================================================
# Demo Functions
# ============================================================================

def demo_basic_distributed_sampler(rank: int, world_size: int):
    """Demonstrate standard DistributedSampler."""
    if rank == 0:
        print("\n" + "=" * 60)
        print("BASIC DISTRIBUTED SAMPLER")
        print("=" * 60)
    
    dataset = SimpleDataset(size=20)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    
    dist.barrier()
    
    # Show indices for each rank
    for r in range(world_size):
        if rank == r:
            sampler.set_epoch(0)
            indices = list(iter(sampler))
            print(f"Rank {rank} indices (epoch 0): {indices}")
        dist.barrier()
    
    # Show different shuffle per epoch
    if rank == 0:
        print("\nWith set_epoch(1):")
    
    dist.barrier()
    
    for r in range(world_size):
        if rank == r:
            sampler.set_epoch(1)
            indices = list(iter(sampler))
            print(f"Rank {rank} indices (epoch 1): {indices}")
        dist.barrier()


def demo_weighted_sampler(rank: int, world_size: int):
    """Demonstrate weighted sampling for imbalanced datasets."""
    if rank == 0:
        print("\n" + "=" * 60)
        print("DISTRIBUTED WEIGHTED SAMPLER")
        print("=" * 60)
    
    dataset = ImbalancedDataset(size=100, num_classes=5)
    
    if rank == 0:
        print(f"Class distribution: {dataset.class_counts.tolist()}")
    
    # Create weights (inverse of class frequency)
    class_weights = 1.0 / dataset.class_counts.float()
    sample_weights = class_weights[dataset.labels]
    
    # Standard sampler (imbalanced)
    std_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    std_loader = DataLoader(dataset, batch_size=10, sampler=std_sampler)
    
    # Weighted sampler (balanced)
    weighted_sampler = DistributedWeightedSampler(
        sample_weights,
        num_samples=len(dataset),
        num_replicas=world_size,
        rank=rank,
    )
    weighted_loader = DataLoader(dataset, batch_size=10, sampler=weighted_sampler)
    
    dist.barrier()
    
    # Compare class distributions
    if rank == 0:
        print("\nComparing class distribution in one epoch:")
        
        # Standard sampler
        std_labels = []
        for batch in std_loader:
            std_labels.extend(batch["label"].tolist())
        std_dist = torch.bincount(torch.tensor(std_labels), minlength=5)
        print(f"Standard sampler: {std_dist.tolist()}")
        
        # Weighted sampler
        weighted_labels = []
        for batch in weighted_loader:
            weighted_labels.extend(batch["label"].tolist())
        weighted_dist = torch.bincount(torch.tensor(weighted_labels), minlength=5)
        print(f"Weighted sampler: {weighted_dist.tolist()}")


def demo_curriculum_sampler(rank: int, world_size: int):
    """Demonstrate curriculum learning sampler."""
    if rank == 0:
        print("\n" + "=" * 60)
        print("DISTRIBUTED CURRICULUM SAMPLER")
        print("=" * 60)
    
    # Create dataset with "difficulty" scores
    dataset_size = 100
    difficulties = torch.rand(dataset_size)  # Random difficulties
    dataset = SimpleDataset(size=dataset_size)
    
    sampler = DistributedCurriculumSampler(
        difficulties,
        num_replicas=world_size,
        rank=rank,
    )
    
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)
    
    if rank == 0:
        print("\nCurriculum progression:")
        
        for progress in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sampler.set_progress(progress)
            sampler.set_epoch(0)
            
            indices = list(iter(sampler))
            avg_difficulty = difficulties[indices].mean().item()
            
            print(f"Progress {progress:.0%}: "
                  f"{len(set(indices))} unique samples, "
                  f"avg difficulty={avg_difficulty:.3f}")


def demo_dataloader_optimization(rank: int, world_size: int):
    """Demonstrate optimized DataLoader configuration."""
    if rank == 0:
        print("\n" + "=" * 60)
        print("OPTIMIZED DATALOADER CONFIGURATION")
        print("=" * 60)
        print("""
Recommended settings for distributed training:

DataLoader(
    dataset,
    batch_size=batch_size,          # Per-GPU batch size
    sampler=dist_sampler,           # NOT shuffle=True
    num_workers=4,                  # 4-8 per GPU typically
    pin_memory=True,                # Faster CPU→GPU transfer
    prefetch_factor=2,              # Batches to prefetch
    persistent_workers=True,        # Keep workers alive
    drop_last=True,                 # Consistent batch sizes
)

Critical reminders:
1. Call sampler.set_epoch(epoch) before each epoch
2. Use non_blocking=True when moving to GPU
3. Ensure total workers * prefetch * batch_size fits in RAM
""")


def main():
    local_rank, rank, world_size = setup_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("DISTRIBUTED SAMPLER PATTERNS")
        print("=" * 60)
        print(f"World size: {world_size}")
    
    # Run demos
    demo_basic_distributed_sampler(rank, world_size)
    demo_weighted_sampler(rank, world_size)
    demo_curriculum_sampler(rank, world_size)
    demo_dataloader_optimization(rank, world_size)
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("DEMOS COMPLETE")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
