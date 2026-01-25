#!/usr/bin/env python3
"""
distributed_benchmarks.py - Comprehensive benchmarking for distributed training

Features:
- Communication bandwidth benchmarks
- Scaling efficiency tests
- Memory usage profiling
- Throughput measurement
- Comparison across configurations

Usage:
    torchrun --nproc_per_node=4 distributed_benchmarks.py
    
    # With custom config
    torchrun --nproc_per_node=8 distributed_benchmarks.py \
        --batch-size 64 \
        --benchmark all

Author: CUDA Lab
"""

import os
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    
    # Test settings
    warmup_iters: int = 10
    benchmark_iters: int = 100
    
    # Communication benchmarks
    message_sizes: List[int] = field(default_factory=lambda: [
        1024,          # 1 KB
        1024 * 1024,   # 1 MB
        10 * 1024 * 1024,  # 10 MB
        100 * 1024 * 1024,  # 100 MB
    ])
    
    # Training benchmarks
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    model_sizes: List[str] = field(default_factory=lambda: ["small", "medium", "large"])
    
    # Output
    output_dir: str = "./benchmark_results"


@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    name: str
    world_size: int
    timestamp: str
    config: Dict
    results: Dict
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================================
# Communication Benchmarks
# ============================================================================

class CommBenchmark:
    """Benchmark distributed communication operations."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
    
    def _measure_time(self, fn, warmup: int = 10, iters: int = 100) -> float:
        """Measure average time for a function."""
        # Warmup
        for _ in range(warmup):
            fn()
        
        torch.cuda.synchronize()
        dist.barrier()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        end = time.perf_counter()
        
        return (end - start) / iters
    
    def benchmark_allreduce(self) -> Dict:
        """Benchmark all-reduce operation."""
        results = {}
        
        for size in self.config.message_sizes:
            tensor = torch.randn(size // 4, device=self.device)  # float32 = 4 bytes
            
            def allreduce():
                dist.all_reduce(tensor)
            
            time_ms = self._measure_time(allreduce) * 1000
            bandwidth_gb = (size * 2 * (self.world_size - 1) / self.world_size) / time_ms / 1e6
            
            results[f"{size // 1024}KB"] = {
                "time_ms": time_ms,
                "bandwidth_GB_s": bandwidth_gb,
            }
            
            if self.rank == 0:
                print(f"  All-reduce {size // 1024} KB: {time_ms:.3f} ms, {bandwidth_gb:.2f} GB/s")
        
        return results
    
    def benchmark_allgather(self) -> Dict:
        """Benchmark all-gather operation."""
        results = {}
        
        for size in self.config.message_sizes:
            tensor = torch.randn(size // 4, device=self.device)
            output = [torch.empty_like(tensor) for _ in range(self.world_size)]
            
            def allgather():
                dist.all_gather(output, tensor)
            
            time_ms = self._measure_time(allgather) * 1000
            bandwidth_gb = (size * (self.world_size - 1)) / time_ms / 1e6
            
            results[f"{size // 1024}KB"] = {
                "time_ms": time_ms,
                "bandwidth_GB_s": bandwidth_gb,
            }
            
            if self.rank == 0:
                print(f"  All-gather {size // 1024} KB: {time_ms:.3f} ms, {bandwidth_gb:.2f} GB/s")
        
        return results
    
    def benchmark_reduce_scatter(self) -> Dict:
        """Benchmark reduce-scatter operation."""
        results = {}
        
        for size in self.config.message_sizes:
            # Input is world_size times larger
            input_tensor = torch.randn(size // 4, device=self.device)
            output = torch.empty(size // 4 // self.world_size, device=self.device)
            input_list = list(input_tensor.chunk(self.world_size))
            
            def reduce_scatter():
                dist.reduce_scatter(output, input_list)
            
            time_ms = self._measure_time(reduce_scatter) * 1000
            bandwidth_gb = (size * (self.world_size - 1) / self.world_size) / time_ms / 1e6
            
            results[f"{size // 1024}KB"] = {
                "time_ms": time_ms,
                "bandwidth_GB_s": bandwidth_gb,
            }
            
            if self.rank == 0:
                print(f"  Reduce-scatter {size // 1024} KB: {time_ms:.3f} ms, {bandwidth_gb:.2f} GB/s")
        
        return results
    
    def benchmark_p2p(self) -> Dict:
        """Benchmark point-to-point communication."""
        results = {}
        
        for size in self.config.message_sizes:
            tensor = torch.randn(size // 4, device=self.device)
            
            def p2p():
                if self.rank == 0:
                    dist.send(tensor, dst=1)
                elif self.rank == 1:
                    dist.recv(tensor, src=0)
                dist.barrier()
            
            if self.world_size >= 2:
                time_ms = self._measure_time(p2p) * 1000
                bandwidth_gb = size / time_ms / 1e6
                
                results[f"{size // 1024}KB"] = {
                    "time_ms": time_ms,
                    "bandwidth_GB_s": bandwidth_gb,
                }
                
                if self.rank == 0:
                    print(f"  P2P {size // 1024} KB: {time_ms:.3f} ms, {bandwidth_gb:.2f} GB/s")
        
        return results
    
    def run_all(self) -> Dict:
        """Run all communication benchmarks."""
        if self.rank == 0:
            print("\n" + "=" * 60)
            print("COMMUNICATION BENCHMARKS")
            print("=" * 60)
        
        results = {}
        
        if self.rank == 0:
            print("\nAll-Reduce:")
        results["allreduce"] = self.benchmark_allreduce()
        
        if self.rank == 0:
            print("\nAll-Gather:")
        results["allgather"] = self.benchmark_allgather()
        
        if self.rank == 0:
            print("\nReduce-Scatter:")
        results["reduce_scatter"] = self.benchmark_reduce_scatter()
        
        if self.world_size >= 2:
            if self.rank == 0:
                print("\nPoint-to-Point (rank 0 <-> rank 1):")
            results["p2p"] = self.benchmark_p2p()
        
        return results


# ============================================================================
# Training Benchmarks
# ============================================================================

def create_model(size: str) -> nn.Module:
    """Create model of specified size."""
    configs = {
        "small": {"hidden": 256, "layers": 4},
        "medium": {"hidden": 512, "layers": 8},
        "large": {"hidden": 1024, "layers": 12},
    }
    
    config = configs[size]
    
    layers = []
    layers.append(nn.Linear(1024, config["hidden"]))
    layers.append(nn.ReLU())
    
    for _ in range(config["layers"]):
        layers.append(nn.Linear(config["hidden"], config["hidden"]))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(config["hidden"], 10))
    
    return nn.Sequential(*layers)


class TrainingBenchmark:
    """Benchmark distributed training throughput."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
    
    def _count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())
    
    def benchmark_throughput(
        self,
        model_size: str,
        batch_size: int,
    ) -> Dict:
        """Benchmark training throughput."""
        # Create model
        model = create_model(model_size).to(self.device)
        model = DDP(model, device_ids=[self.rank])
        
        num_params = self._count_parameters(model)
        
        # Create dummy data
        X = torch.randn(batch_size * 10, 1024, device=self.device)
        y = torch.randint(0, 10, (batch_size * 10,), device=self.device)
        
        dataset = TensorDataset(X, y)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        
        # Setup
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(self.config.warmup_iters):
            for batch in dataloader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                break
        
        torch.cuda.synchronize()
        dist.barrier()
        
        # Benchmark
        total_samples = 0
        start = time.perf_counter()
        
        for i in range(self.config.benchmark_iters):
            for batch in dataloader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                total_samples += batch_size * self.world_size
                break
        
        torch.cuda.synchronize()
        dist.barrier()
        end = time.perf_counter()
        
        # Calculate metrics
        elapsed = end - start
        samples_per_sec = total_samples / elapsed
        time_per_step_ms = elapsed / self.config.benchmark_iters * 1000
        
        # Memory usage
        torch.cuda.reset_peak_memory_stats()
        memory_gb = torch.cuda.max_memory_allocated() / 1e9
        
        return {
            "model_size": model_size,
            "num_parameters": num_params,
            "batch_size": batch_size,
            "global_batch_size": batch_size * self.world_size,
            "samples_per_sec": samples_per_sec,
            "time_per_step_ms": time_per_step_ms,
            "memory_gb": memory_gb,
        }
    
    def benchmark_scaling(self, model_size: str, batch_size: int) -> Dict:
        """Measure scaling efficiency (requires multiple runs with different world sizes)."""
        result = self.benchmark_throughput(model_size, batch_size)
        
        # Scaling efficiency would require baseline from single GPU
        # For now, just return throughput
        return result
    
    def run_all(self) -> Dict:
        """Run all training benchmarks."""
        if self.rank == 0:
            print("\n" + "=" * 60)
            print("TRAINING BENCHMARKS")
            print("=" * 60)
        
        results = []
        
        for model_size in self.config.model_sizes:
            for batch_size in self.config.batch_sizes:
                if self.rank == 0:
                    print(f"\nModel: {model_size}, Batch size: {batch_size}")
                
                try:
                    result = self.benchmark_throughput(model_size, batch_size)
                    results.append(result)
                    
                    if self.rank == 0:
                        print(f"  Throughput: {result['samples_per_sec']:.0f} samples/sec")
                        print(f"  Step time: {result['time_per_step_ms']:.2f} ms")
                        print(f"  Memory: {result['memory_gb']:.2f} GB")
                
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if self.rank == 0:
                            print(f"  OOM - skipping")
                        torch.cuda.empty_cache()
                    else:
                        raise
        
        return {"training_results": results}


# ============================================================================
# Memory Benchmarks
# ============================================================================

class MemoryBenchmark:
    """Benchmark memory usage patterns."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{self.rank}")
    
    def benchmark_activation_memory(self, model_size: str, batch_size: int) -> Dict:
        """Measure activation memory during forward pass."""
        model = create_model(model_size).to(self.device)
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        baseline = torch.cuda.memory_allocated()
        
        X = torch.randn(batch_size, 1024, device=self.device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(X)
        
        peak = torch.cuda.max_memory_allocated()
        activation_memory = (peak - baseline) / 1e9
        
        return {
            "model_size": model_size,
            "batch_size": batch_size,
            "activation_memory_gb": activation_memory,
        }
    
    def benchmark_gradient_memory(self, model_size: str) -> Dict:
        """Measure memory for gradients."""
        model = create_model(model_size).to(self.device)
        
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        grad_memory = param_memory  # Same size as parameters
        
        return {
            "model_size": model_size,
            "param_memory_gb": param_memory,
            "grad_memory_gb": grad_memory,
            "total_gb": param_memory + grad_memory,
        }
    
    def run_all(self) -> Dict:
        """Run all memory benchmarks."""
        if self.rank == 0:
            print("\n" + "=" * 60)
            print("MEMORY BENCHMARKS")
            print("=" * 60)
        
        results = {
            "activation_memory": [],
            "gradient_memory": [],
        }
        
        for model_size in self.config.model_sizes:
            # Gradient memory
            grad_result = self.benchmark_gradient_memory(model_size)
            results["gradient_memory"].append(grad_result)
            
            if self.rank == 0:
                print(f"\n{model_size} model:")
                print(f"  Parameters: {grad_result['param_memory_gb']:.3f} GB")
                print(f"  Gradients: {grad_result['grad_memory_gb']:.3f} GB")
            
            # Activation memory for different batch sizes
            for batch_size in self.config.batch_sizes[:2]:  # Limit to avoid OOM
                try:
                    act_result = self.benchmark_activation_memory(model_size, batch_size)
                    results["activation_memory"].append(act_result)
                    
                    if self.rank == 0:
                        print(f"  Activations (bs={batch_size}): {act_result['activation_memory_gb']:.3f} GB")
                except RuntimeError:
                    if self.rank == 0:
                        print(f"  Activations (bs={batch_size}): OOM")
                    torch.cuda.empty_cache()
        
        return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Distributed training benchmarks")
    parser.add_argument("--benchmark", type=str, default="all",
                       choices=["all", "comm", "training", "memory"],
                       help="Which benchmark to run")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./benchmark_results")
    args = parser.parse_args()
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print("=" * 60)
        print("DISTRIBUTED TRAINING BENCHMARKS")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name(local_rank)}")
    
    # Config
    config = BenchmarkConfig(
        warmup_iters=args.warmup,
        benchmark_iters=args.iters,
        output_dir=args.output_dir,
    )
    
    all_results = {}
    
    # Run benchmarks
    if args.benchmark in ["all", "comm"]:
        comm_bench = CommBenchmark(config)
        all_results["communication"] = comm_bench.run_all()
    
    if args.benchmark in ["all", "training"]:
        train_bench = TrainingBenchmark(config)
        all_results["training"] = train_bench.run_all()
    
    if args.benchmark in ["all", "memory"]:
        mem_bench = MemoryBenchmark(config)
        all_results["memory"] = mem_bench.run_all()
    
    # Save results
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        
        result = BenchmarkResult(
            name=f"benchmark_{world_size}gpu",
            world_size=world_size,
            timestamp=datetime.now().isoformat(),
            config={
                "warmup_iters": config.warmup_iters,
                "benchmark_iters": config.benchmark_iters,
                "device": torch.cuda.get_device_name(local_rank),
            },
            results=all_results,
        )
        
        output_path = os.path.join(
            config.output_dir,
            f"benchmark_{world_size}gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        result.save(output_path)
        
        print("\n" + "=" * 60)
        print(f"Results saved to: {output_path}")
        print("=" * 60)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
