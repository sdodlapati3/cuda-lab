"""
matmul/benchmark.py - Matrix Multiplication Kernel Benchmark

Compares different GEMM implementations:
1. PyTorch native (cuBLAS)
2. Naive CUDA kernel
3. Tiled CUDA kernel
4. Tensor Core (if available)

Author: CUDA Lab
"""

import torch
import time
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import subprocess
import os

# Add parent to path for template
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from templates.benchmark_template import KernelBenchmark, BenchmarkResult


@dataclass
class MatMulConfig:
    """Configuration for matrix multiplication benchmark."""
    M: int  # Rows of A
    N: int  # Cols of B
    K: int  # Cols of A / Rows of B
    dtype: str = 'float32'
    
    @property
    def flops(self) -> int:
        """FLOPs for GEMM: 2*M*N*K (multiply-add)."""
        return 2 * self.M * self.N * self.K
    
    @property
    def bytes_transferred(self) -> int:
        """Minimum bytes: read A, B; write C."""
        elem_size = 4 if self.dtype == 'float32' else 2
        return elem_size * (self.M * self.K + self.K * self.N + self.M * self.N)


class MatMulBenchmark(KernelBenchmark):
    """Benchmark suite for matrix multiplication kernels."""
    
    def __init__(self, device: str = 'cuda:0'):
        super().__init__(
            name="MatMul Benchmark",
            description="Compare GEMM implementations across sizes and precisions"
        )
        self.device = torch.device(device)
        self.results: List[Dict] = []
        
    def setup(self, config: MatMulConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create input matrices."""
        dtype = torch.float32 if config.dtype == 'float32' else torch.float16
        
        A = torch.randn(config.M, config.K, dtype=dtype, device=self.device)
        B = torch.randn(config.K, config.N, dtype=dtype, device=self.device)
        
        return A, B
    
    def pytorch_matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """PyTorch native matmul (uses cuBLAS)."""
        return torch.matmul(A, B)
    
    def pytorch_mm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """PyTorch mm (explicit 2D matrix multiply)."""
        return torch.mm(A, B)
    
    def pytorch_bmm(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Batched matrix multiply (single batch)."""
        return torch.bmm(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)
    
    def einsum_matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Einstein summation notation."""
        return torch.einsum('ik,kj->ij', A, B)
    
    def benchmark_kernel(
        self,
        kernel_fn,
        A: torch.Tensor,
        B: torch.Tensor,
        warmup: int = 10,
        iterations: int = 100
    ) -> BenchmarkResult:
        """Benchmark a single kernel function."""
        # Warmup
        for _ in range(warmup):
            _ = kernel_fn(A, B)
        
        torch.cuda.synchronize()
        
        # Timed iterations
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        
        for i in range(iterations):
            start_events[i].record()
            _ = kernel_fn(A, B)
            end_events[i].record()
        
        torch.cuda.synchronize()
        
        # Collect timings
        times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iterations)]
        
        return BenchmarkResult(
            kernel_name=kernel_fn.__name__,
            mean_time_ms=sum(times_ms) / len(times_ms),
            std_time_ms=torch.tensor(times_ms).std().item(),
            min_time_ms=min(times_ms),
            max_time_ms=max(times_ms),
            iterations=iterations
        )
    
    def run_size_sweep(
        self,
        sizes: List[int] = None,
        dtype: str = 'float32'
    ) -> List[Dict]:
        """Run benchmark across matrix sizes."""
        if sizes is None:
            sizes = [256, 512, 1024, 2048, 4096, 8192]
        
        results = []
        
        for size in sizes:
            config = MatMulConfig(M=size, N=size, K=size, dtype=dtype)
            A, B = self.setup(config)
            
            print(f"\nBenchmarking {size}x{size}x{size} ({dtype})")
            print("-" * 50)
            
            # Benchmark each implementation
            kernels = [
                ('torch.matmul', self.pytorch_matmul),
                ('torch.mm', self.pytorch_mm),
                ('torch.einsum', self.einsum_matmul),
            ]
            
            for name, kernel_fn in kernels:
                try:
                    result = self.benchmark_kernel(kernel_fn, A, B)
                    
                    # Calculate TFLOPS
                    tflops = config.flops / (result.mean_time_ms * 1e-3) / 1e12
                    
                    result_dict = {
                        'size': size,
                        'dtype': dtype,
                        'kernel': name,
                        'mean_ms': result.mean_time_ms,
                        'std_ms': result.std_time_ms,
                        'tflops': tflops,
                        'arithmetic_intensity': config.flops / config.bytes_transferred,
                    }
                    results.append(result_dict)
                    
                    print(f"  {name:20s}: {result.mean_time_ms:8.3f} ms  "
                          f"({tflops:6.2f} TFLOPS)")
                    
                except Exception as e:
                    print(f"  {name:20s}: ERROR - {e}")
            
            # Clean up
            del A, B
            torch.cuda.empty_cache()
        
        return results
    
    def run_precision_comparison(self, size: int = 4096) -> List[Dict]:
        """Compare FP32 vs FP16 (with/without Tensor Cores)."""
        results = []
        
        for dtype in ['float32', 'float16']:
            config = MatMulConfig(M=size, N=size, K=size, dtype=dtype)
            A, B = self.setup(config)
            
            print(f"\nPrecision: {dtype} @ {size}x{size}")
            print("-" * 50)
            
            result = self.benchmark_kernel(self.pytorch_matmul, A, B)
            tflops = config.flops / (result.mean_time_ms * 1e-3) / 1e12
            
            # Check if Tensor Cores are being used (FP16 should be much faster)
            results.append({
                'size': size,
                'dtype': dtype,
                'mean_ms': result.mean_time_ms,
                'tflops': tflops,
            })
            
            print(f"  torch.matmul: {result.mean_time_ms:.3f} ms ({tflops:.2f} TFLOPS)")
            
            del A, B
            torch.cuda.empty_cache()
        
        # Calculate speedup
        if len(results) == 2:
            speedup = results[0]['mean_ms'] / results[1]['mean_ms']
            print(f"\nFP16 speedup: {speedup:.2f}x")
        
        return results
    
    def run_batch_benchmark(
        self,
        batch_sizes: List[int] = None,
        matrix_size: int = 1024
    ) -> List[Dict]:
        """Benchmark batched matrix multiplication."""
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32, 64]
        
        results = []
        
        print(f"\nBatched MatMul @ {matrix_size}x{matrix_size}")
        print("-" * 50)
        
        for batch_size in batch_sizes:
            A = torch.randn(batch_size, matrix_size, matrix_size, 
                           dtype=torch.float32, device=self.device)
            B = torch.randn(batch_size, matrix_size, matrix_size,
                           dtype=torch.float32, device=self.device)
            
            # Warmup
            for _ in range(10):
                _ = torch.bmm(A, B)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(100):
                _ = torch.bmm(A, B)
            end.record()
            
            torch.cuda.synchronize()
            mean_ms = start.elapsed_time(end) / 100
            
            flops = 2 * batch_size * matrix_size ** 3
            tflops = flops / (mean_ms * 1e-3) / 1e12
            
            results.append({
                'batch_size': batch_size,
                'matrix_size': matrix_size,
                'mean_ms': mean_ms,
                'tflops': tflops,
            })
            
            print(f"  Batch {batch_size:3d}: {mean_ms:8.3f} ms ({tflops:6.2f} TFLOPS)")
            
            del A, B
            torch.cuda.empty_cache()
        
        return results
    
    def verify_correctness(self, size: int = 1024) -> bool:
        """Verify all implementations produce correct results."""
        print(f"\nVerifying correctness @ {size}x{size}")
        print("-" * 50)
        
        A = torch.randn(size, size, dtype=torch.float32, device=self.device)
        B = torch.randn(size, size, dtype=torch.float32, device=self.device)
        
        # Reference result
        ref = torch.mm(A, B)
        
        # Test each implementation
        tests = [
            ('torch.matmul', self.pytorch_matmul(A, B)),
            ('torch.einsum', self.einsum_matmul(A, B)),
            ('torch.bmm', self.pytorch_bmm(A, B)),
        ]
        
        all_passed = True
        for name, result in tests:
            max_diff = (result - ref).abs().max().item()
            passed = max_diff < 1e-4
            all_passed = all_passed and passed
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name:20s}: {status} (max diff: {max_diff:.2e})")
        
        return all_passed
    
    def export_results(self, results: List[Dict], output_path: str):
        """Export results to JSON for roofline plotting."""
        # Convert to roofline format
        roofline_data = {
            'kernels': [
                {
                    'name': f"{r['kernel']} ({r['size']}x{r['size']})",
                    'arithmetic_intensity': r.get('arithmetic_intensity', r['size'] / 3),
                    'performance': r['tflops'],
                }
                for r in results if 'kernel' in r
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(roofline_data, f, indent=2)
        
        print(f"\nExported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='MatMul Benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', 
                       default=[512, 1024, 2048, 4096],
                       help='Matrix sizes to benchmark')
    parser.add_argument('--dtype', type=str, default='float32',
                       choices=['float32', 'float16'])
    parser.add_argument('--output', type=str, default='matmul_results.json')
    parser.add_argument('--verify', action='store_true', help='Verify correctness')
    parser.add_argument('--precision-test', action='store_true',
                       help='Compare FP32 vs FP16')
    parser.add_argument('--batch-test', action='store_true',
                       help='Test batched matmul')
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    benchmark = MatMulBenchmark()
    
    if args.verify:
        benchmark.verify_correctness()
    
    # Run size sweep
    results = benchmark.run_size_sweep(args.sizes, args.dtype)
    
    if args.precision_test:
        precision_results = benchmark.run_precision_comparison()
        results.extend(precision_results)
    
    if args.batch_test:
        batch_results = benchmark.run_batch_benchmark()
        results.extend(batch_results)
    
    # Export results
    benchmark.export_results(results, args.output)


if __name__ == "__main__":
    main()
