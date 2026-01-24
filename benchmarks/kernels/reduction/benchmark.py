"""
reduction_benchmark.py - Benchmark various reduction implementations

This benchmark compares:
1. Naive reduction (baseline)
2. Sequential addressing
3. First add during load
4. Warp-level primitives (__shfl_down_sync)
5. CUB library (reference)
6. PyTorch (reference)

Usage:
    python reduction_benchmark.py
    python reduction_benchmark.py --size 1000000
"""

import torch
import time
import argparse
from typing import Callable, Dict, List, Tuple
import json

# Check for custom CUDA implementations
try:
    import reduction_cuda
    HAS_CUSTOM_CUDA = True
except ImportError:
    HAS_CUSTOM_CUDA = False


class ReductionBenchmark:
    """Benchmark suite for reduction operations."""
    
    def __init__(
        self,
        sizes: List[int] = None,
        warmup: int = 10,
        iterations: int = 100
    ):
        self.sizes = sizes or [2**i for i in range(10, 25)]  # 1K to 16M
        self.warmup = warmup
        self.iterations = iterations
        self.results: Dict[str, Dict[int, float]] = {}
        
        # Device info
        self.device = torch.device('cuda')
        self.device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        self.peak_bandwidth = props.memory_clock_rate * 2 * props.memory_bus_width / 8 / 1e6  # GB/s
        
        print(f"Device: {self.device_name}")
        print(f"Peak memory bandwidth: ~{self.peak_bandwidth:.0f} GB/s")
    
    def time_kernel(self, fn: Callable, *args) -> float:
        """Time a kernel with warmup and iterations."""
        # Warmup
        for _ in range(self.warmup):
            fn(*args)
        torch.cuda.synchronize()
        
        # Timed iterations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(self.iterations):
            fn(*args)
        end.record()
        
        torch.cuda.synchronize()
        return start.elapsed_time(end) / self.iterations
    
    def pytorch_sum(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch reduction (reference)."""
        return torch.sum(x)
    
    def benchmark_impl(self, name: str, fn: Callable, sizes: List[int] = None):
        """Benchmark a single implementation across sizes."""
        sizes = sizes or self.sizes
        self.results[name] = {}
        
        for n in sizes:
            x = torch.randn(n, device=self.device)
            
            try:
                time_ms = self.time_kernel(fn, x)
                
                # Calculate bandwidth
                bytes_moved = n * 4  # Input only (output is scalar)
                bandwidth = bytes_moved / (time_ms * 1e6)  # GB/s
                
                self.results[name][n] = {
                    'time_ms': time_ms,
                    'bandwidth_GB_s': bandwidth,
                    'pct_peak': bandwidth / self.peak_bandwidth * 100
                }
                
            except Exception as e:
                print(f"  {name} failed at n={n}: {e}")
                self.results[name][n] = {'error': str(e)}
    
    def run_all(self):
        """Run all benchmarks."""
        print("\nRunning benchmarks...")
        
        # PyTorch sum (reference)
        print("  PyTorch sum...")
        self.benchmark_impl("pytorch_sum", self.pytorch_sum)
        
        # Custom CUDA implementations (if available)
        if HAS_CUSTOM_CUDA:
            print("  Custom CUDA naive...")
            self.benchmark_impl("naive", reduction_cuda.naive_reduce)
            
            print("  Custom CUDA optimized...")
            self.benchmark_impl("optimized", reduction_cuda.optimized_reduce)
            
            print("  Custom CUDA warp shuffle...")
            self.benchmark_impl("warp_shuffle", reduction_cuda.warp_shuffle_reduce)
    
    def print_results(self):
        """Print results as table."""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{'='*80}")
        print(f"Reduction Benchmark Results - {self.device_name}")
        print(f"{'='*80}")
        
        # Get all sizes that have results
        all_sizes = sorted(set(
            size for impl_results in self.results.values() 
            for size in impl_results.keys()
        ))
        
        # Print header
        impl_names = list(self.results.keys())
        header = f"{'Size':>12}"
        for name in impl_names:
            header += f" {name:>15}"
        print(header)
        print("-" * len(header))
        
        # Print rows
        for size in all_sizes:
            row = f"{size:>12,}"
            for name in impl_names:
                if size in self.results[name]:
                    result = self.results[name][size]
                    if 'bandwidth_GB_s' in result:
                        row += f" {result['bandwidth_GB_s']:>12.1f} GB/s"
                    else:
                        row += f" {'ERROR':>15}"
                else:
                    row += f" {'-':>15}"
            print(row)
        
        print("=" * 80)
        
        # Print efficiency summary
        print("\nEfficiency (% of peak bandwidth):")
        for name in impl_names:
            if self.results[name]:
                # Get efficiency at largest size
                largest_size = max(self.results[name].keys())
                result = self.results[name][largest_size]
                if 'pct_peak' in result:
                    print(f"  {name}: {result['pct_peak']:.1f}%")
    
    def save_results(self, path: str):
        """Save results to JSON."""
        data = {
            'device': self.device_name,
            'peak_bandwidth_GB_s': self.peak_bandwidth,
            'warmup': self.warmup,
            'iterations': self.iterations,
            'results': self.results
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Reduction Benchmark')
    parser.add_argument('--size', type=int, default=None,
                       help='Single size to benchmark')
    parser.add_argument('--output', type=str, default='reduction_results.json',
                       help='Output file for results')
    args = parser.parse_args()
    
    if args.size:
        sizes = [args.size]
    else:
        sizes = [2**i for i in range(10, 25)]  # 1K to 16M
    
    benchmark = ReductionBenchmark(sizes=sizes)
    benchmark.run_all()
    benchmark.print_results()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
