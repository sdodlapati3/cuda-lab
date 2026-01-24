"""
benchmark_template.py - Reusable template for CUDA kernel benchmarking

Features:
- Accurate GPU timing with CUDA events
- Warmup iterations
- Statistical analysis (mean, std, percentiles)
- Comparison against baselines
- JSON export for reproducibility

Usage:
    benchmark = KernelBenchmark("my_kernel")
    time_ms = benchmark.time_kernel(my_kernel_fn, *args)
    benchmark.add_result("my_impl", time_ms, bytes_moved=N*4*3)
    benchmark.print_results()
    benchmark.save("results.json")
"""

import torch
import json
import time
from pathlib import Path
from typing import Callable, Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from statistics import mean, stdev


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str = ""  # or kernel_name
    time_ms: float = 0.0  # mean time (also: mean_time_ms)
    time_std_ms: float = 0.0  # std deviation (also: std_time_ms)
    bandwidth_GB_s: Optional[float] = None
    tflops: Optional[float] = None
    percent_of_peak: Optional[float] = None
    min_time_ms: Optional[float] = None
    max_time_ms: Optional[float] = None
    iterations: Optional[int] = None
    
    # Aliases for compatibility
    @property
    def kernel_name(self) -> str:
        return self.name
    
    @property
    def mean_time_ms(self) -> float:
        return self.time_ms
    
    @property
    def std_time_ms(self) -> float:
        return self.time_std_ms


class KernelBenchmark:
    """
    GPU kernel benchmarking utility.
    
    Example:
        benchmark = KernelBenchmark("reduction")
        
        # Time a kernel
        def run_naive():
            naive_reduce(d_input, d_output, n)
        
        time_ms = benchmark.time_kernel(run_naive, iterations=100)
        benchmark.add_result("naive", time_ms, bytes_moved=n * 4)
        
        # Compare implementations
        benchmark.print_results()
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        warmup: int = 10,
        iterations: int = 100,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            name: Benchmark name
            description: Benchmark description
            warmup: Number of warmup iterations
            iterations: Number of timed iterations
            device: CUDA device (default: current device)
        """
        self.name = name
        self.description = description
        self.warmup = warmup
        self.iterations = iterations
        self.device = device or torch.device('cuda')
        self.results: Dict[str, BenchmarkResult] = {}
        
        # Get device info
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_props = torch.cuda.get_device_properties(self.device)
        
        # Peak theoretical performance
        # Note: Adjust these for your GPU
        self.peak_bandwidth_GB_s = self._estimate_peak_bandwidth()
        self.peak_tflops_fp32 = self._estimate_peak_tflops()
    
    def _estimate_peak_bandwidth(self) -> float:
        """Estimate peak memory bandwidth based on device."""
        # Memory bandwidth = clock * bus_width * 2 (DDR) / 8 (bits to bytes)
        # These are rough estimates - replace with actual specs
        bandwidth_map = {
            'A100': 2039,
            'H100': 3350,
            'V100': 900,
            'T4': 320,
            'RTX 3090': 936,
            'RTX 4090': 1008,
        }
        
        for gpu_name, bw in bandwidth_map.items():
            if gpu_name in self.device_name:
                return bw
        
        # Default estimate
        return 500.0
    
    def _estimate_peak_tflops(self) -> float:
        """Estimate peak FP32 TFLOPS based on device."""
        tflops_map = {
            'A100': 19.5,
            'H100': 67.0,
            'V100': 15.7,
            'T4': 8.1,
            'RTX 3090': 35.6,
            'RTX 4090': 82.6,
        }
        
        for gpu_name, tf in tflops_map.items():
            if gpu_name in self.device_name:
                return tf
        
        return 10.0
    
    def time_kernel(
        self,
        fn: Callable,
        *args,
        **kwargs
    ) -> List[float]:
        """
        Time a kernel function.
        
        Args:
            fn: Function to benchmark (should launch CUDA kernel)
            *args, **kwargs: Arguments to pass to fn
            
        Returns:
            List of timing results (ms) for each iteration
        """
        # Warmup
        for _ in range(self.warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        
        # Timed iterations
        times = []
        for _ in range(self.iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            fn(*args, **kwargs)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        return times
    
    def add_result(
        self,
        name: str,
        times: List[float],
        bytes_moved: int = 0,
        flops: int = 0,
    ):
        """
        Add benchmark result.
        
        Args:
            name: Implementation name
            times: List of timing results (ms)
            bytes_moved: Total bytes moved (for bandwidth calc)
            flops: Total FLOPs (for TFLOPS calc)
        """
        time_ms = mean(times)
        time_std = stdev(times) if len(times) > 1 else 0.0
        
        bandwidth = None
        if bytes_moved > 0:
            bandwidth = bytes_moved / (time_ms * 1e6)  # GB/s
        
        tflops = None
        if flops > 0:
            tflops = flops / (time_ms * 1e9)  # TFLOPS
        
        # Calculate percent of peak
        percent = None
        if bandwidth is not None:
            percent = 100 * bandwidth / self.peak_bandwidth_GB_s
        elif tflops is not None:
            percent = 100 * tflops / self.peak_tflops_fp32
        
        self.results[name] = BenchmarkResult(
            name=name,
            time_ms=time_ms,
            time_std_ms=time_std,
            bandwidth_GB_s=bandwidth,
            tflops=tflops,
            percent_of_peak=percent,
        )
    
    def print_results(self, sort_by: str = 'time_ms'):
        """Print benchmark results as formatted table."""
        if not self.results:
            print("No results to display.")
            return
        
        print(f"\n{'='*70}")
        print(f"Benchmark: {self.name}")
        print(f"Device: {self.device_name}")
        print(f"Peak BW: {self.peak_bandwidth_GB_s:.0f} GB/s, Peak FP32: {self.peak_tflops_fp32:.1f} TFLOPS")
        print(f"{'='*70}")
        
        # Sort results
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: getattr(x, sort_by) or float('inf')
        )
        
        # Print header
        has_bandwidth = any(r.bandwidth_GB_s is not None for r in sorted_results)
        has_tflops = any(r.tflops is not None for r in sorted_results)
        
        header = f"{'Implementation':<20} {'Time (ms)':>12} {'Std':>8}"
        if has_bandwidth:
            header += f" {'BW (GB/s)':>12} {'% Peak':>8}"
        if has_tflops:
            header += f" {'TFLOPS':>10} {'% Peak':>8}"
        
        print(header)
        print("-" * len(header))
        
        # Print results
        for result in sorted_results:
            row = f"{result.name:<20} {result.time_ms:>12.3f} {result.time_std_ms:>8.3f}"
            if has_bandwidth:
                bw = result.bandwidth_GB_s or 0
                pct = result.percent_of_peak or 0 if result.bandwidth_GB_s else 0
                row += f" {bw:>12.1f} {pct:>7.1f}%"
            if has_tflops:
                tf = result.tflops or 0
                pct = result.percent_of_peak or 0 if result.tflops else 0
                row += f" {tf:>10.2f} {pct:>7.1f}%"
            print(row)
        
        print(f"{'='*70}\n")
    
    def save(self, path: str):
        """Save results to JSON file."""
        data = {
            'benchmark_name': self.name,
            'device': self.device_name,
            'peak_bandwidth_GB_s': self.peak_bandwidth_GB_s,
            'peak_tflops_fp32': self.peak_tflops_fp32,
            'warmup': self.warmup,
            'iterations': self.iterations,
            'results': {name: asdict(result) for name, result in self.results.items()},
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'KernelBenchmark':
        """Load benchmark results from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        benchmark = cls(data['benchmark_name'])
        benchmark.peak_bandwidth_GB_s = data.get('peak_bandwidth_GB_s', 500)
        benchmark.peak_tflops_fp32 = data.get('peak_tflops_fp32', 10)
        
        for name, result_data in data.get('results', {}).items():
            benchmark.results[name] = BenchmarkResult(**result_data)
        
        return benchmark


def compare_benchmarks(benchmarks: List[KernelBenchmark], metric: str = 'time_ms'):
    """Compare results across multiple benchmark runs."""
    print(f"\n{'='*80}")
    print("Benchmark Comparison")
    print(f"{'='*80}")
    
    # Collect all implementation names
    all_names = set()
    for bm in benchmarks:
        all_names.update(bm.results.keys())
    
    # Print comparison
    header = f"{'Implementation':<20}"
    for bm in benchmarks:
        header += f" {bm.name:>15}"
    print(header)
    print("-" * len(header))
    
    for name in sorted(all_names):
        row = f"{name:<20}"
        for bm in benchmarks:
            if name in bm.results:
                val = getattr(bm.results[name], metric)
                row += f" {val:>15.3f}"
            else:
                row += f" {'N/A':>15}"
        print(row)
    
    print(f"{'='*80}\n")


# Example usage
if __name__ == "__main__":
    import torch
    
    # Simple vector add benchmark
    N = 1 << 24  # 16M elements
    
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    c = torch.empty(N, device='cuda')
    
    benchmark = KernelBenchmark("vector_add", warmup=10, iterations=100)
    
    # Benchmark PyTorch add
    def pytorch_add():
        torch.add(a, b, out=c)
    
    times = benchmark.time_kernel(pytorch_add)
    benchmark.add_result(
        "pytorch",
        times,
        bytes_moved=N * 4 * 3  # Read a, read b, write c
    )
    
    benchmark.print_results()
    benchmark.save("vectoradd_results.json")
