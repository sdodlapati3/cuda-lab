"""
scaling_benchmark.py - Measure strong/weak scaling for distributed workloads

Purpose:
- Strong scaling: Fixed problem size, increase GPUs
- Weak scaling: Problem size proportional to GPUs
- Communication overhead analysis

Required Environment Variables:
- MASTER_ADDR: IP of rank 0 node
- MASTER_PORT: Port for communication
- WORLD_SIZE: Total number of processes
- RANK: Global rank of this process
- LOCAL_RANK: Local rank on this node

Usage:
    # Single-node multi-GPU
    torchrun --nproc_per_node=4 scaling_benchmark.py --test strong
    
    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \\
             --master_addr=<ip> --master_port=29500 \\
             scaling_benchmark.py --test weak
"""

import os
import time
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        # Single GPU fallback
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


class ScalingBenchmark:
    """
    Measure strong and weak scaling efficiency.
    
    Strong Scaling: Fixed total problem size
        - Ideal: Time reduces linearly with GPUs
        - Efficiency = T(1) / (N * T(N))
        
    Weak Scaling: Problem size per GPU is fixed
        - Ideal: Time stays constant
        - Efficiency = T(1) / T(N)
    """
    
    def __init__(
        self,
        name: str,
        rank: int,
        world_size: int,
        warmup: int = 5,
        iterations: int = 20,
    ):
        self.name = name
        self.rank = rank
        self.world_size = world_size
        self.warmup = warmup
        self.iterations = iterations
        self.results: Dict[str, float] = {}
        
        if is_main_process():
            print(f"ScalingBenchmark: {name}")
            print(f"World size: {world_size}")
            print(f"Warmup: {warmup}, Iterations: {iterations}")
    
    def time_operation(self, fn, *args, **kwargs) -> float:
        """
        Time an operation with proper synchronization.
        
        Returns:
            Average time in milliseconds
        """
        # Warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        
        # Barrier to sync all processes
        if dist.is_initialized():
            dist.barrier()
        
        # Timed iterations
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(self.iterations):
            fn(*args, **kwargs)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        # Sync and collect times from all ranks
        local_time = (end - start) / self.iterations * 1000  # ms
        
        if dist.is_initialized():
            # Get max time across all ranks (bottleneck determines throughput)
            time_tensor = torch.tensor([local_time], device='cuda')
            dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
            return time_tensor.item()
        
        return local_time
    
    def benchmark_allreduce(self, size_mb: float = 100) -> Dict[str, float]:
        """
        Benchmark allreduce communication.
        
        Args:
            size_mb: Size of tensor to allreduce in MB
        """
        n_elements = int(size_mb * 1024 * 1024 / 4)  # float32
        tensor = torch.randn(n_elements, device='cuda')
        
        def allreduce_op():
            if dist.is_initialized():
                dist.all_reduce(tensor)
        
        time_ms = self.time_operation(allreduce_op)
        
        # Calculate bandwidth
        # AllReduce moves 2*(N-1)/N * size data per GPU
        data_moved = 2 * (self.world_size - 1) / self.world_size * size_mb * 1024
        bandwidth = data_moved / time_ms  # GB/s
        
        results = {
            'operation': 'allreduce',
            'size_mb': size_mb,
            'time_ms': time_ms,
            'bandwidth_GB_s': bandwidth,
            'world_size': self.world_size,
        }
        
        if is_main_process():
            print(f"AllReduce {size_mb:.0f} MB: {time_ms:.2f} ms, {bandwidth:.1f} GB/s")
        
        return results
    
    def benchmark_allgather(self, size_mb: float = 100) -> Dict[str, float]:
        """Benchmark allgather communication."""
        n_elements = int(size_mb * 1024 * 1024 / 4)
        tensor = torch.randn(n_elements, device='cuda')
        output_tensors = [torch.empty(n_elements, device='cuda') 
                         for _ in range(self.world_size)]
        
        def allgather_op():
            if dist.is_initialized():
                dist.all_gather(output_tensors, tensor)
        
        time_ms = self.time_operation(allgather_op)
        
        # AllGather moves (N-1)/N * size data per GPU
        data_moved = (self.world_size - 1) * size_mb * 1024
        bandwidth = data_moved / time_ms
        
        results = {
            'operation': 'allgather',
            'size_mb': size_mb,
            'time_ms': time_ms,
            'bandwidth_GB_s': bandwidth,
            'world_size': self.world_size,
        }
        
        if is_main_process():
            print(f"AllGather {size_mb:.0f} MB: {time_ms:.2f} ms, {bandwidth:.1f} GB/s")
        
        return results
    
    def benchmark_matmul_strong_scaling(
        self,
        total_size: int = 16384,
    ) -> Dict[str, float]:
        """
        Strong scaling benchmark for matrix multiplication.
        
        Distributes columns across GPUs.
        """
        # Each GPU handles a portion of columns
        local_cols = total_size // self.world_size
        
        A = torch.randn(total_size, total_size, device='cuda')
        B = torch.randn(total_size, local_cols, device='cuda')
        
        def matmul_op():
            C = torch.matmul(A, B)
            # AllGather to combine results
            if dist.is_initialized():
                C_list = [torch.empty_like(C) for _ in range(self.world_size)]
                dist.all_gather(C_list, C)
        
        time_ms = self.time_operation(matmul_op)
        
        # FLOPs for full matmul
        total_flops = 2 * total_size * total_size * total_size
        tflops = total_flops / (time_ms * 1e9)
        
        results = {
            'test': 'strong_scaling',
            'operation': 'matmul',
            'matrix_size': total_size,
            'time_ms': time_ms,
            'tflops': tflops,
            'world_size': self.world_size,
        }
        
        if is_main_process():
            print(f"Strong Scaling MatMul {total_size}x{total_size}: "
                  f"{time_ms:.2f} ms, {tflops:.2f} TFLOPS")
        
        return results
    
    def benchmark_matmul_weak_scaling(
        self,
        size_per_gpu: int = 8192,
    ) -> Dict[str, float]:
        """
        Weak scaling benchmark for matrix multiplication.
        
        Each GPU handles same-sized local problem.
        """
        A = torch.randn(size_per_gpu, size_per_gpu, device='cuda')
        B = torch.randn(size_per_gpu, size_per_gpu, device='cuda')
        
        def matmul_op():
            C = torch.matmul(A, B)
        
        time_ms = self.time_operation(matmul_op)
        
        # FLOPs per GPU
        local_flops = 2 * size_per_gpu * size_per_gpu * size_per_gpu
        tflops = local_flops / (time_ms * 1e9)
        
        # Total throughput
        total_tflops = tflops * self.world_size
        
        results = {
            'test': 'weak_scaling',
            'operation': 'matmul',
            'size_per_gpu': size_per_gpu,
            'time_ms': time_ms,
            'tflops_per_gpu': tflops,
            'total_tflops': total_tflops,
            'world_size': self.world_size,
        }
        
        if is_main_process():
            print(f"Weak Scaling MatMul {size_per_gpu}x{size_per_gpu}/GPU: "
                  f"{time_ms:.2f} ms, {tflops:.2f} TFLOPS/GPU, "
                  f"{total_tflops:.2f} Total TFLOPS")
        
        return results
    
    def benchmark_ddp_forward(
        self,
        model_size: str = 'small',
        batch_size_per_gpu: int = 32,
    ) -> Dict[str, float]:
        """
        Benchmark DDP forward pass.
        
        Args:
            model_size: 'small', 'medium', or 'large'
            batch_size_per_gpu: Batch size per GPU
        """
        # Create model
        sizes = {
            'small': (512, 256, 128),
            'medium': (2048, 1024, 512),
            'large': (4096, 2048, 1024),
        }
        hidden_sizes = sizes.get(model_size, sizes['small'])
        
        model = torch.nn.Sequential(
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_sizes[2], 10),
        ).cuda()
        
        if dist.is_initialized():
            model = DDP(model)
        
        input_data = torch.randn(batch_size_per_gpu, hidden_sizes[0], device='cuda')
        
        def forward_op():
            output = model(input_data)
        
        time_ms = self.time_operation(forward_op)
        
        # Throughput
        samples_per_sec = batch_size_per_gpu * self.world_size / (time_ms / 1000)
        
        results = {
            'test': 'ddp_forward',
            'model_size': model_size,
            'batch_size_per_gpu': batch_size_per_gpu,
            'time_ms': time_ms,
            'samples_per_sec': samples_per_sec,
            'world_size': self.world_size,
        }
        
        if is_main_process():
            print(f"DDP Forward ({model_size}): {time_ms:.2f} ms, "
                  f"{samples_per_sec:.0f} samples/sec")
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save benchmark results to JSON."""
        if not is_main_process():
            return
        
        data = {
            'benchmark_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'world_size': self.world_size,
            'device': torch.cuda.get_device_name(0),
            'results': results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {output_path}")


def run_communication_benchmark(benchmark: ScalingBenchmark) -> List[Dict]:
    """Run communication benchmarks."""
    results = []
    
    if is_main_process():
        print("\n=== Communication Benchmarks ===")
    
    for size_mb in [1, 10, 100, 500]:
        results.append(benchmark.benchmark_allreduce(size_mb))
        results.append(benchmark.benchmark_allgather(size_mb))
    
    return results


def run_strong_scaling_benchmark(benchmark: ScalingBenchmark) -> List[Dict]:
    """Run strong scaling benchmarks."""
    results = []
    
    if is_main_process():
        print("\n=== Strong Scaling Benchmarks ===")
    
    for size in [4096, 8192, 16384]:
        results.append(benchmark.benchmark_matmul_strong_scaling(size))
    
    return results


def run_weak_scaling_benchmark(benchmark: ScalingBenchmark) -> List[Dict]:
    """Run weak scaling benchmarks."""
    results = []
    
    if is_main_process():
        print("\n=== Weak Scaling Benchmarks ===")
    
    for size in [4096, 8192]:
        results.append(benchmark.benchmark_matmul_weak_scaling(size))
    
    return results


def run_ddp_benchmark(benchmark: ScalingBenchmark) -> List[Dict]:
    """Run DDP benchmarks."""
    results = []
    
    if is_main_process():
        print("\n=== DDP Benchmarks ===")
    
    for model_size in ['small', 'medium', 'large']:
        results.append(benchmark.benchmark_ddp_forward(model_size))
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'comm', 'strong', 'weak', 'ddp'])
    parser.add_argument('--output', type=str, default='scaling_results.json')
    args = parser.parse_args()
    
    rank, local_rank, world_size = setup_distributed()
    
    try:
        benchmark = ScalingBenchmark(
            name=f"scaling_{args.test}",
            rank=rank,
            world_size=world_size,
        )
        
        results = []
        
        if args.test in ['all', 'comm']:
            results.extend(run_communication_benchmark(benchmark))
        
        if args.test in ['all', 'strong']:
            results.extend(run_strong_scaling_benchmark(benchmark))
        
        if args.test in ['all', 'weak']:
            results.extend(run_weak_scaling_benchmark(benchmark))
        
        if args.test in ['all', 'ddp']:
            results.extend(run_ddp_benchmark(benchmark))
        
        benchmark.save_results(results, args.output)
        
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
