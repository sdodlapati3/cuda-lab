"""
softmax/benchmark.py - Softmax Kernel Benchmark

Compares different softmax implementations:
1. PyTorch native
2. PyTorch functional
3. Manual (numerically stable)
4. Flash-attention style fused

Author: CUDA Lab
"""

import torch
import torch.nn.functional as F
import time
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from templates.benchmark_template import KernelBenchmark, BenchmarkResult


@dataclass
class SoftmaxConfig:
    """Configuration for softmax benchmark."""
    batch_size: int
    seq_length: int
    hidden_dim: int
    dtype: str = 'float32'
    
    @property
    def elements(self) -> int:
        return self.batch_size * self.seq_length * self.hidden_dim
    
    @property
    def flops(self) -> int:
        """Approximate FLOPs: exp, sum, div per element."""
        # Per row: hidden_dim exps + hidden_dim sums + hidden_dim divs
        rows = self.batch_size * self.seq_length
        return rows * (3 * self.hidden_dim)  # Simplified estimate
    
    @property
    def bytes_transferred(self) -> int:
        """Read input, write output."""
        elem_size = 4 if self.dtype == 'float32' else 2
        return 2 * self.elements * elem_size


class SoftmaxBenchmark(KernelBenchmark):
    """Benchmark suite for softmax implementations."""
    
    def __init__(self, device: str = 'cuda:0'):
        super().__init__(
            name="Softmax Benchmark",
            description="Compare softmax implementations for transformer workloads"
        )
        self.device = torch.device(device)
        
    def setup(self, config: SoftmaxConfig) -> torch.Tensor:
        """Create input tensor."""
        dtype = torch.float32 if config.dtype == 'float32' else torch.float16
        x = torch.randn(
            config.batch_size, config.seq_length, config.hidden_dim,
            dtype=dtype, device=self.device
        )
        return x
    
    def pytorch_softmax_module(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """PyTorch nn.Softmax."""
        softmax = torch.nn.Softmax(dim=dim)
        return softmax(x)
    
    def pytorch_softmax_functional(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """PyTorch F.softmax."""
        return F.softmax(x, dim=dim)
    
    def manual_softmax_stable(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Numerically stable softmax implementation."""
        x_max = x.max(dim=dim, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)
    
    def fused_softmax_scale(
        self, 
        x: torch.Tensor, 
        scale: float = 1.0,
        dim: int = -1
    ) -> torch.Tensor:
        """Fused scale + softmax (common in attention)."""
        return F.softmax(x * scale, dim=dim)
    
    def online_softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Online softmax - single pass for max and sum.
        This is what FlashAttention uses internally.
        """
        # Note: This is a simulation - actual online softmax is in CUDA
        # PyTorch doesn't expose online softmax directly
        return F.softmax(x, dim=dim)
    
    def log_softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Log softmax (numerically stable for cross-entropy)."""
        return F.log_softmax(x, dim=dim)
    
    def benchmark_kernel(
        self,
        kernel_fn,
        x: torch.Tensor,
        warmup: int = 10,
        iterations: int = 100,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a single kernel."""
        # Warmup
        for _ in range(warmup):
            _ = kernel_fn(x, **kwargs)
        
        torch.cuda.synchronize()
        
        # Timed run
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        
        for i in range(iterations):
            start_events[i].record()
            _ = kernel_fn(x, **kwargs)
            end_events[i].record()
        
        torch.cuda.synchronize()
        
        times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(iterations)]
        
        return BenchmarkResult(
            kernel_name=kernel_fn.__name__,
            mean_time_ms=sum(times_ms) / len(times_ms),
            std_time_ms=torch.tensor(times_ms).std().item(),
            min_time_ms=min(times_ms),
            max_time_ms=max(times_ms),
            iterations=iterations
        )
    
    def run_sequence_sweep(
        self,
        seq_lengths: List[int] = None,
        batch_size: int = 32,
        hidden_dim: int = 768,
        dtype: str = 'float32'
    ) -> List[Dict]:
        """Benchmark across sequence lengths (important for transformers)."""
        if seq_lengths is None:
            seq_lengths = [128, 256, 512, 1024, 2048, 4096]
        
        results = []
        
        for seq_len in seq_lengths:
            config = SoftmaxConfig(
                batch_size=batch_size,
                seq_length=seq_len,
                hidden_dim=hidden_dim,
                dtype=dtype
            )
            x = self.setup(config)
            
            print(f"\nSeq Length: {seq_len} (batch={batch_size}, hidden={hidden_dim})")
            print("-" * 60)
            
            kernels = [
                ('F.softmax', self.pytorch_softmax_functional),
                ('manual_stable', self.manual_softmax_stable),
                ('F.log_softmax', self.log_softmax),
            ]
            
            for name, kernel_fn in kernels:
                try:
                    result = self.benchmark_kernel(kernel_fn, x)
                    
                    # Calculate bandwidth (GB/s)
                    bandwidth_GBs = config.bytes_transferred / (result.mean_time_ms * 1e-3) / 1e9
                    
                    result_dict = {
                        'seq_length': seq_len,
                        'batch_size': batch_size,
                        'hidden_dim': hidden_dim,
                        'dtype': dtype,
                        'kernel': name,
                        'mean_ms': result.mean_time_ms,
                        'std_ms': result.std_time_ms,
                        'bandwidth_GBs': bandwidth_GBs,
                        'elements_M': config.elements / 1e6,
                    }
                    results.append(result_dict)
                    
                    print(f"  {name:20s}: {result.mean_time_ms:8.4f} ms "
                          f"({bandwidth_GBs:7.1f} GB/s)")
                    
                except Exception as e:
                    print(f"  {name:20s}: ERROR - {e}")
            
            del x
            torch.cuda.empty_cache()
        
        return results
    
    def run_attention_pattern(
        self,
        batch_size: int = 8,
        num_heads: int = 12,
        seq_lengths: List[int] = None
    ) -> List[Dict]:
        """
        Benchmark softmax in attention pattern.
        Shape: (batch, heads, seq, seq)
        """
        if seq_lengths is None:
            seq_lengths = [128, 256, 512, 1024]
        
        results = []
        
        print(f"\nAttention Softmax (batch={batch_size}, heads={num_heads})")
        print("-" * 60)
        
        for seq_len in seq_lengths:
            # Attention scores shape: (B, H, S, S)
            scores = torch.randn(
                batch_size, num_heads, seq_len, seq_len,
                dtype=torch.float32, device=self.device
            )
            
            # Benchmark
            result = self.benchmark_kernel(
                self.pytorch_softmax_functional, 
                scores,
                dim=-1
            )
            
            # Memory: read + write attention matrix
            bytes_total = 2 * scores.numel() * 4  # float32
            bandwidth_GBs = bytes_total / (result.mean_time_ms * 1e-3) / 1e9
            
            results.append({
                'pattern': 'attention',
                'seq_length': seq_len,
                'batch_size': batch_size,
                'num_heads': num_heads,
                'mean_ms': result.mean_time_ms,
                'bandwidth_GBs': bandwidth_GBs,
                'memory_MB': scores.numel() * 4 / 1e6,
            })
            
            print(f"  Seq {seq_len:4d}: {result.mean_time_ms:8.4f} ms "
                  f"({bandwidth_GBs:7.1f} GB/s) "
                  f"[{scores.numel() * 4 / 1e6:.1f} MB]")
            
            del scores
            torch.cuda.empty_cache()
        
        return results
    
    def verify_correctness(self, config: SoftmaxConfig = None) -> bool:
        """Verify all implementations match."""
        if config is None:
            config = SoftmaxConfig(batch_size=4, seq_length=128, hidden_dim=512)
        
        print(f"\nVerifying correctness")
        print("-" * 60)
        
        x = self.setup(config)
        
        # Reference
        ref = F.softmax(x, dim=-1)
        
        tests = [
            ('nn.Softmax', self.pytorch_softmax_module(x)),
            ('manual_stable', self.manual_softmax_stable(x)),
            ('log_softmax', torch.exp(self.log_softmax(x))),  # exp to compare
        ]
        
        all_passed = True
        for name, result in tests:
            max_diff = (result - ref).abs().max().item()
            passed = max_diff < 1e-5
            all_passed = all_passed and passed
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name:20s}: {status} (max diff: {max_diff:.2e})")
        
        # Also verify softmax properties
        print("\n  Property checks:")
        sums = ref.sum(dim=-1)
        sum_check = (sums - 1.0).abs().max().item() < 1e-5
        print(f"    Sum to 1:         {'✓ PASS' if sum_check else '✗ FAIL'}")
        
        positive_check = (ref >= 0).all().item()
        print(f"    All positive:     {'✓ PASS' if positive_check else '✗ FAIL'}")
        
        return all_passed and sum_check and positive_check
    
    def export_results(self, results: List[Dict], output_path: str):
        """Export results to JSON."""
        with open(output_path, 'w') as f:
            json.dump({'results': results}, f, indent=2)
        print(f"\nExported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Softmax Benchmark')
    parser.add_argument('--seq-lengths', type=int, nargs='+',
                       default=[256, 512, 1024, 2048])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=768)
    parser.add_argument('--attention-test', action='store_true',
                       help='Test attention softmax pattern')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=str, default='softmax_results.json')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    benchmark = SoftmaxBenchmark()
    
    if args.verify:
        benchmark.verify_correctness()
    
    results = benchmark.run_sequence_sweep(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim
    )
    
    if args.attention_test:
        attn_results = benchmark.run_attention_pattern()
        results.extend(attn_results)
    
    benchmark.export_results(results, args.output)


if __name__ == "__main__":
    main()
