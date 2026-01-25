#!/usr/bin/env python3
"""
ddp_benchmark.py - Measure DDP scaling efficiency

Tests linear scaling with different GPU counts and configurations.

Usage:
    # Test scaling from 1 to 8 GPUs
    for n in 1 2 4 8; do
        torchrun --nproc_per_node=$n ddp_benchmark.py --gpus $n
    done

Author: CUDA Lab
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler


@dataclass
class BenchmarkResult:
    """Benchmark result for a single configuration."""
    num_gpus: int
    batch_size_per_gpu: int
    effective_batch_size: int
    samples_per_sec: float
    time_per_step_ms: float
    scaling_efficiency: float  # Compared to single GPU baseline
    gpu_memory_mb: float
    

def create_model(model_name: str = "resnet50") -> nn.Module:
    """Create model for benchmarking."""
    import torchvision.models as models
    
    model_dict = {
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "vit_b_16": models.vit_b_16,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")
    
    return model_dict[model_name](num_classes=1000)


def create_synthetic_data(batch_size: int, num_batches: int = 100) -> TensorDataset:
    """Create synthetic data for benchmarking."""
    # ImageNet-like dimensions
    total_samples = batch_size * num_batches
    images = torch.randn(total_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (total_samples,))
    return TensorDataset(images, labels)


def setup_distributed():
    """Initialize distributed environment."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        torch.cuda.set_device(0)
    
    return local_rank, rank, world_size


def benchmark_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    warmup_steps: int = 10,
    benchmark_steps: int = 50,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Benchmark training throughput."""
    model.train()
    
    # Get batch size from first batch
    batch_size = next(iter(dataloader))[0].size(0)
    
    # Warmup
    data_iter = iter(dataloader)
    for _ in range(warmup_steps):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)
        
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    
    for step in range(benchmark_steps):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)
        
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time = end_time - start_time
    time_per_step = total_time / benchmark_steps * 1000  # ms
    samples_per_sec = batch_size * benchmark_steps / total_time
    
    # GPU memory
    gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    return {
        'time_per_step_ms': time_per_step,
        'samples_per_sec': samples_per_sec,
        'gpu_memory_mb': gpu_memory_mb,
    }


def run_benchmark(
    model_name: str = "resnet50",
    batch_size: int = 32,
    use_amp: bool = True,
    warmup_steps: int = 10,
    benchmark_steps: int = 50,
) -> BenchmarkResult:
    """Run benchmark with current GPU configuration."""
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Create model
    model = create_model(model_name).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create data
    dataset = create_synthetic_data(batch_size, num_batches=100)
    
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=4, pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = GradScaler() if use_amp else None
    
    # Run benchmark
    results = benchmark_throughput(
        model, dataloader, criterion, optimizer, scaler, device,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        use_amp=use_amp,
    )
    
    # Aggregate across processes
    if world_size > 1:
        # Get total throughput
        throughput_tensor = torch.tensor([results['samples_per_sec']], device=device)
        dist.all_reduce(throughput_tensor, op=dist.ReduceOp.SUM)
        results['samples_per_sec'] = throughput_tensor.item()
        
        # Get max memory
        memory_tensor = torch.tensor([results['gpu_memory_mb']], device=device)
        dist.all_reduce(memory_tensor, op=dist.ReduceOp.MAX)
        results['gpu_memory_mb'] = memory_tensor.item()
    
    effective_batch_size = batch_size * world_size
    
    return BenchmarkResult(
        num_gpus=world_size,
        batch_size_per_gpu=batch_size,
        effective_batch_size=effective_batch_size,
        samples_per_sec=results['samples_per_sec'],
        time_per_step_ms=results['time_per_step_ms'],
        scaling_efficiency=1.0,  # Will be calculated later
        gpu_memory_mb=results['gpu_memory_mb'],
    )


def print_results(results: List[BenchmarkResult], baseline_throughput: float):
    """Print benchmark results in a nice table."""
    print("\n" + "=" * 80)
    print("DDP SCALING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'GPUs':>6} {'Batch/GPU':>10} {'Eff Batch':>10} {'Samples/s':>12} "
          f"{'ms/step':>10} {'Scaling %':>10} {'Memory MB':>10}")
    print("-" * 80)
    
    for r in results:
        # Calculate scaling efficiency
        ideal_throughput = baseline_throughput * r.num_gpus
        scaling_efficiency = (r.samples_per_sec / ideal_throughput) * 100
        
        print(f"{r.num_gpus:>6} {r.batch_size_per_gpu:>10} {r.effective_batch_size:>10} "
              f"{r.samples_per_sec:>12.1f} {r.time_per_step_ms:>10.2f} "
              f"{scaling_efficiency:>9.1f}% {r.gpu_memory_mb:>10.0f}")
    
    print("=" * 80)
    print("\nScaling efficiency = (actual throughput) / (ideal throughput) × 100%")
    print("Ideal throughput = single GPU throughput × number of GPUs")


def main():
    parser = argparse.ArgumentParser(description='DDP Scaling Benchmark')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152', 'vit_b_16'])
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision')
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--benchmark-steps', type=int, default=50)
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()
    
    result = run_benchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
    )
    
    # Only rank 0 prints and saves
    local_rank, rank, world_size = setup_distributed() if 'RANK' not in os.environ else (
        int(os.environ['LOCAL_RANK']), dist.get_rank(), dist.get_world_size()
    )
    
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK RESULTS - {args.model.upper()}")
        print(f"{'=' * 60}")
        print(f"Configuration:")
        print(f"  Model: {args.model}")
        print(f"  GPUs: {result.num_gpus}")
        print(f"  Batch size per GPU: {result.batch_size_per_gpu}")
        print(f"  Effective batch size: {result.effective_batch_size}")
        print(f"  Mixed precision: {not args.no_amp}")
        print(f"\nResults:")
        print(f"  Throughput: {result.samples_per_sec:.1f} samples/sec")
        print(f"  Time per step: {result.time_per_step_ms:.2f} ms")
        print(f"  Peak GPU memory: {result.gpu_memory_mb:.0f} MB")
        print(f"{'=' * 60}\n")
        
        # Save to file if requested
        if args.output:
            result_dict = {
                'model': args.model,
                'num_gpus': result.num_gpus,
                'batch_size_per_gpu': result.batch_size_per_gpu,
                'effective_batch_size': result.effective_batch_size,
                'samples_per_sec': result.samples_per_sec,
                'time_per_step_ms': result.time_per_step_ms,
                'gpu_memory_mb': result.gpu_memory_mb,
                'use_amp': not args.no_amp,
            }
            with open(args.output, 'w') as f:
                json.dump(result_dict, f, indent=2)
            print(f"Results saved to {args.output}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
