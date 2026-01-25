#!/usr/bin/env python3
"""
bandwidth_test.py - NCCL bandwidth and latency benchmarking

Measures:
- All-reduce bandwidth and throughput
- All-gather bandwidth
- Point-to-point bandwidth
- Latency for small messages

Usage:
    torchrun --nproc_per_node=4 bandwidth_test.py
    torchrun --nproc_per_node=4 bandwidth_test.py --sizes 1M,10M,100M,1G
    
    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=4 bandwidth_test.py

Author: CUDA Lab
"""

import os
import time
import argparse
from typing import List, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    operation: str
    size_bytes: int
    time_ms: float
    bandwidth_gbps: float
    algbw_gbps: float  # Algorithm bandwidth


def parse_size(size_str: str) -> int:
    """Parse size string like '1M', '100K', '1G' to bytes."""
    size_str = size_str.upper().strip()
    multipliers = {
        'K': 1024,
        'M': 1024 ** 2,
        'G': 1024 ** 3,
    }
    
    if size_str[-1] in multipliers:
        return int(float(size_str[:-1]) * multipliers[size_str[-1]])
    return int(size_str)


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def sync_and_time() -> float:
    """Synchronize CUDA and return time."""
    torch.cuda.synchronize()
    dist.barrier()
    return time.perf_counter()


def benchmark_all_reduce(
    sizes: List[int],
    warmup: int = 5,
    iterations: int = 20,
    rank: int = 0,
    world_size: int = 1,
) -> List[BenchmarkResult]:
    """Benchmark all-reduce operation."""
    results = []
    
    for size in sizes:
        # Create tensor
        num_elements = size // 4  # float32 = 4 bytes
        tensor = torch.randn(num_elements, dtype=torch.float32).cuda()
        
        # Warmup
        for _ in range(warmup):
            dist.all_reduce(tensor)
        
        # Benchmark
        torch.cuda.synchronize()
        dist.barrier()
        
        times = []
        for _ in range(iterations):
            start = sync_and_time()
            dist.all_reduce(tensor)
            end = sync_and_time()
            times.append((end - start) * 1000)  # ms
        
        avg_time = sum(times) / len(times)
        
        # Calculate bandwidth
        # All-reduce transfers 2*(N-1)/N * data_size in ring algorithm
        algo_bytes = 2 * (world_size - 1) / world_size * size
        bus_bandwidth = algo_bytes / (avg_time / 1000) / 1e9  # GB/s
        
        # Also calculate "algorithm bandwidth" (data_size / time)
        alg_bandwidth = size / (avg_time / 1000) / 1e9
        
        results.append(BenchmarkResult(
            operation="all_reduce",
            size_bytes=size,
            time_ms=avg_time,
            bandwidth_gbps=bus_bandwidth * 8,  # Convert to Gbps
            algbw_gbps=alg_bandwidth * 8,
        ))
        
        if rank == 0:
            print(f"  {size/1e6:8.2f} MB | {avg_time:8.3f} ms | "
                  f"{bus_bandwidth:8.2f} GB/s | {alg_bandwidth:8.2f} GB/s (algbw)")
    
    return results


def benchmark_all_gather(
    sizes: List[int],
    warmup: int = 5,
    iterations: int = 20,
    rank: int = 0,
    world_size: int = 1,
) -> List[BenchmarkResult]:
    """Benchmark all-gather operation."""
    results = []
    
    for size in sizes:
        # Each rank contributes size/world_size bytes
        local_size = size // world_size
        num_elements = local_size // 4
        
        tensor = torch.randn(num_elements, dtype=torch.float32).cuda()
        output = [torch.zeros(num_elements, dtype=torch.float32).cuda() 
                 for _ in range(world_size)]
        
        # Warmup
        for _ in range(warmup):
            dist.all_gather(output, tensor)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = sync_and_time()
            dist.all_gather(output, tensor)
            end = sync_and_time()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        
        # All-gather transfers (N-1)/N * data_size
        algo_bytes = (world_size - 1) / world_size * size
        bus_bandwidth = algo_bytes / (avg_time / 1000) / 1e9
        alg_bandwidth = size / (avg_time / 1000) / 1e9
        
        results.append(BenchmarkResult(
            operation="all_gather",
            size_bytes=size,
            time_ms=avg_time,
            bandwidth_gbps=bus_bandwidth * 8,
            algbw_gbps=alg_bandwidth * 8,
        ))
        
        if rank == 0:
            print(f"  {size/1e6:8.2f} MB | {avg_time:8.3f} ms | "
                  f"{bus_bandwidth:8.2f} GB/s | {alg_bandwidth:8.2f} GB/s (algbw)")
    
    return results


def benchmark_reduce_scatter(
    sizes: List[int],
    warmup: int = 5,
    iterations: int = 20,
    rank: int = 0,
    world_size: int = 1,
) -> List[BenchmarkResult]:
    """Benchmark reduce-scatter operation."""
    results = []
    
    for size in sizes:
        num_elements = size // 4
        # Input is world_size chunks, output is 1 chunk
        input_tensor = torch.randn(num_elements, dtype=torch.float32).cuda()
        output = torch.zeros(num_elements // world_size, dtype=torch.float32).cuda()
        
        # Warmup
        for _ in range(warmup):
            dist.reduce_scatter_tensor(output, input_tensor)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = sync_and_time()
            dist.reduce_scatter_tensor(output, input_tensor)
            end = sync_and_time()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        
        algo_bytes = (world_size - 1) / world_size * size
        bus_bandwidth = algo_bytes / (avg_time / 1000) / 1e9
        alg_bandwidth = size / (avg_time / 1000) / 1e9
        
        results.append(BenchmarkResult(
            operation="reduce_scatter",
            size_bytes=size,
            time_ms=avg_time,
            bandwidth_gbps=bus_bandwidth * 8,
            algbw_gbps=alg_bandwidth * 8,
        ))
        
        if rank == 0:
            print(f"  {size/1e6:8.2f} MB | {avg_time:8.3f} ms | "
                  f"{bus_bandwidth:8.2f} GB/s | {alg_bandwidth:8.2f} GB/s (algbw)")
    
    return results


def benchmark_broadcast(
    sizes: List[int],
    warmup: int = 5,
    iterations: int = 20,
    rank: int = 0,
    world_size: int = 1,
) -> List[BenchmarkResult]:
    """Benchmark broadcast operation."""
    results = []
    
    for size in sizes:
        num_elements = size // 4
        tensor = torch.randn(num_elements, dtype=torch.float32).cuda()
        
        # Warmup
        for _ in range(warmup):
            dist.broadcast(tensor, src=0)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = sync_and_time()
            dist.broadcast(tensor, src=0)
            end = sync_and_time()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        
        bus_bandwidth = size / (avg_time / 1000) / 1e9
        
        results.append(BenchmarkResult(
            operation="broadcast",
            size_bytes=size,
            time_ms=avg_time,
            bandwidth_gbps=bus_bandwidth * 8,
            algbw_gbps=bus_bandwidth * 8,
        ))
        
        if rank == 0:
            print(f"  {size/1e6:8.2f} MB | {avg_time:8.3f} ms | "
                  f"{bus_bandwidth:8.2f} GB/s")
    
    return results


def benchmark_p2p(
    sizes: List[int],
    warmup: int = 5,
    iterations: int = 20,
    rank: int = 0,
    world_size: int = 1,
) -> List[BenchmarkResult]:
    """Benchmark point-to-point send/recv."""
    if world_size < 2:
        return []
    
    results = []
    
    for size in sizes:
        num_elements = size // 4
        
        if rank == 0:
            tensor = torch.randn(num_elements, dtype=torch.float32).cuda()
        else:
            tensor = torch.zeros(num_elements, dtype=torch.float32).cuda()
        
        # Only rank 0 and 1 participate
        if rank <= 1:
            # Warmup
            for _ in range(warmup):
                if rank == 0:
                    dist.send(tensor, dst=1)
                else:
                    dist.recv(tensor, src=0)
            
            # Benchmark
            times = []
            for _ in range(iterations):
                torch.cuda.synchronize()
                if rank == 0:
                    start = time.perf_counter()
                    dist.send(tensor, dst=1)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                else:
                    start = time.perf_counter()
                    dist.recv(tensor, src=0)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                times.append((end - start) * 1000)
        
        dist.barrier()
        
        if rank <= 1:
            avg_time = sum(times) / len(times)
            bandwidth = size / (avg_time / 1000) / 1e9
            
            if rank == 0:
                results.append(BenchmarkResult(
                    operation="send_recv",
                    size_bytes=size,
                    time_ms=avg_time,
                    bandwidth_gbps=bandwidth * 8,
                    algbw_gbps=bandwidth * 8,
                ))
                print(f"  {size/1e6:8.2f} MB | {avg_time:8.3f} ms | "
                      f"{bandwidth:8.2f} GB/s")
    
    return results


def benchmark_latency(
    rank: int,
    world_size: int,
    iterations: int = 100,
) -> float:
    """Measure base latency with tiny messages."""
    tensor = torch.zeros(1, dtype=torch.float32).cuda()
    
    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor)
    
    # Measure
    times = []
    for _ in range(iterations):
        start = sync_and_time()
        dist.all_reduce(tensor)
        end = sync_and_time()
        times.append((end - start) * 1e6)  # microseconds
    
    return sum(times) / len(times)


def print_summary(all_results: Dict[str, List[BenchmarkResult]], rank: int):
    """Print summary of all benchmarks."""
    if rank != 0:
        return
    
    print("\n" + "=" * 70)
    print("BANDWIDTH SUMMARY")
    print("=" * 70)
    print(f"{'Operation':<20} {'Best BW (GB/s)':<18} {'Best Size':<15}")
    print("-" * 70)
    
    for op, results in all_results.items():
        if results:
            best = max(results, key=lambda r: r.bandwidth_gbps)
            print(f"{op:<20} {best.bandwidth_gbps/8:>12.2f} GB/s    "
                  f"{best.size_bytes/1e6:>8.2f} MB")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="NCCL Bandwidth Test")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--sizes", type=str, default="1M,10M,100M,500M,1G",
                       help="Comma-separated sizes to test (e.g., 1M,10M,100M)")
    parser.add_argument("--ops", type=str, default="all",
                       help="Operations to test: all,allreduce,allgather,reducescatter,broadcast,p2p")
    args = parser.parse_args()
    
    # Parse sizes
    sizes = [parse_size(s) for s in args.sizes.split(",")]
    
    # Setup
    local_rank, rank, world_size = setup_distributed()
    
    if rank == 0:
        print("=" * 70)
        print("NCCL BANDWIDTH BENCHMARK")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
        print(f"Sizes: {[f'{s/1e6:.1f}MB' for s in sizes]}")
        print("=" * 70)
    
    all_results = {}
    ops_to_run = args.ops.split(",") if args.ops != "all" else \
                 ["allreduce", "allgather", "reducescatter", "broadcast", "p2p", "latency"]
    
    # Latency test
    if "latency" in ops_to_run or "all" in ops_to_run:
        if rank == 0:
            print("\n--- LATENCY TEST ---")
        
        latency = benchmark_latency(rank, world_size)
        
        if rank == 0:
            print(f"Base latency (1 element all-reduce): {latency:.2f} µs")
    
    # All-Reduce
    if "allreduce" in ops_to_run or "all" in ops_to_run:
        if rank == 0:
            print("\n--- ALL-REDUCE BENCHMARK ---")
            print(f"{'Size':>12} | {'Time':>10} | {'Bus BW':>10} | {'Alg BW':>12}")
            print("-" * 55)
        
        all_results["all_reduce"] = benchmark_all_reduce(
            sizes, args.warmup, args.iterations, rank, world_size
        )
    
    # All-Gather
    if "allgather" in ops_to_run or "all" in ops_to_run:
        if rank == 0:
            print("\n--- ALL-GATHER BENCHMARK ---")
            print(f"{'Size':>12} | {'Time':>10} | {'Bus BW':>10} | {'Alg BW':>12}")
            print("-" * 55)
        
        all_results["all_gather"] = benchmark_all_gather(
            sizes, args.warmup, args.iterations, rank, world_size
        )
    
    # Reduce-Scatter
    if "reducescatter" in ops_to_run or "all" in ops_to_run:
        if rank == 0:
            print("\n--- REDUCE-SCATTER BENCHMARK ---")
            print(f"{'Size':>12} | {'Time':>10} | {'Bus BW':>10} | {'Alg BW':>12}")
            print("-" * 55)
        
        all_results["reduce_scatter"] = benchmark_reduce_scatter(
            sizes, args.warmup, args.iterations, rank, world_size
        )
    
    # Broadcast
    if "broadcast" in ops_to_run or "all" in ops_to_run:
        if rank == 0:
            print("\n--- BROADCAST BENCHMARK ---")
            print(f"{'Size':>12} | {'Time':>10} | {'Bus BW':>10}")
            print("-" * 45)
        
        all_results["broadcast"] = benchmark_broadcast(
            sizes, args.warmup, args.iterations, rank, world_size
        )
    
    # Point-to-point
    if "p2p" in ops_to_run or "all" in ops_to_run:
        if rank == 0:
            print("\n--- POINT-TO-POINT BENCHMARK (Rank 0 → Rank 1) ---")
            print(f"{'Size':>12} | {'Time':>10} | {'BW':>10}")
            print("-" * 40)
        
        all_results["send_recv"] = benchmark_p2p(
            sizes, args.warmup, args.iterations, rank, world_size
        )
    
    # Summary
    print_summary(all_results, rank)
    
    if rank == 0:
        print("\nBenchmark complete!")
        print("\nTips:")
        print("- For NVLink systems, expect 200-300 GB/s per link")
        print("- For PCIe Gen4 x16, expect ~25 GB/s")
        print("- For InfiniBand HDR, expect ~25 GB/s")
        print("- Small message latency is typically 5-50 µs")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
