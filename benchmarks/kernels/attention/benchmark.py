"""
attention/benchmark.py - Self-Attention Kernel Benchmark

Compares attention implementations:
1. Standard scaled dot-product attention
2. PyTorch 2.0 scaled_dot_product_attention (Flash/Memory-efficient)
3. Manual implementation
4. xFormers (if available)

Author: CUDA Lab
"""

import torch
import torch.nn.functional as F
import math
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from templates.benchmark_template import KernelBenchmark, BenchmarkResult


@dataclass
class AttentionConfig:
    """Configuration for attention benchmark."""
    batch_size: int
    num_heads: int
    seq_length: int
    head_dim: int
    dtype: str = 'float32'
    causal: bool = False
    
    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_dim
    
    @property
    def flops(self) -> int:
        """
        Attention FLOPs estimate:
        - Q @ K^T: 2 * B * H * S * S * D
        - Softmax: ~5 * B * H * S * S
        - Attn @ V: 2 * B * H * S * S * D
        """
        B, H, S, D = self.batch_size, self.num_heads, self.seq_length, self.head_dim
        qk_flops = 2 * B * H * S * S * D
        softmax_flops = 5 * B * H * S * S
        av_flops = 2 * B * H * S * S * D
        return qk_flops + softmax_flops + av_flops
    
    @property
    def memory_standard(self) -> int:
        """Standard attention memory (stores full attention matrix)."""
        B, H, S, D = self.batch_size, self.num_heads, self.seq_length, self.head_dim
        elem_size = 4 if self.dtype == 'float32' else 2
        # Q, K, V inputs + attention matrix + output
        qkv = 3 * B * H * S * D * elem_size
        attn_matrix = B * H * S * S * elem_size
        output = B * H * S * D * elem_size
        return qkv + attn_matrix + output
    
    @property
    def memory_flash(self) -> int:
        """Flash attention memory (no full attention matrix)."""
        B, H, S, D = self.batch_size, self.num_heads, self.seq_length, self.head_dim
        elem_size = 4 if self.dtype == 'float32' else 2
        # Q, K, V inputs + output (no attention matrix stored)
        return 4 * B * H * S * D * elem_size


class AttentionBenchmark(KernelBenchmark):
    """Benchmark suite for attention mechanisms."""
    
    def __init__(self, device: str = 'cuda:0'):
        super().__init__(
            name="Attention Benchmark",
            description="Compare attention implementations for transformers"
        )
        self.device = torch.device(device)
        self._check_backends()
    
    def _check_backends(self):
        """Check available attention backends."""
        self.has_flash = hasattr(F, 'scaled_dot_product_attention')
        self.has_xformers = False
        
        try:
            import xformers.ops as xops
            self.has_xformers = True
            self.xops = xops
        except ImportError:
            pass
        
        print(f"Flash Attention (PyTorch 2.0): {'✓' if self.has_flash else '✗'}")
        print(f"xFormers: {'✓' if self.has_xformers else '✗'}")
    
    def setup(self, config: AttentionConfig) -> Tuple[torch.Tensor, ...]:
        """Create Q, K, V tensors."""
        dtype = torch.float32 if config.dtype == 'float32' else torch.float16
        
        # Shape: (batch, heads, seq, head_dim)
        Q = torch.randn(
            config.batch_size, config.num_heads, config.seq_length, config.head_dim,
            dtype=dtype, device=self.device
        )
        K = torch.randn_like(Q)
        V = torch.randn_like(Q)
        
        return Q, K, V
    
    def standard_attention(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        scale = 1.0 / math.sqrt(Q.size(-1))
        
        # Q @ K^T
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Causal mask
        if causal:
            seq_len = Q.size(-2)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Attention @ V
        return torch.matmul(attn_weights, V)
    
    def sdpa_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor, 
        V: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """PyTorch 2.0 scaled_dot_product_attention."""
        return F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=causal,
            dropout_p=0.0
        )
    
    def sdpa_flash_only(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """Force Flash Attention backend."""
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False
        ):
            return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    
    def sdpa_mem_efficient(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """Force Memory-efficient attention backend."""
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=True
        ):
            return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    
    def xformers_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """xFormers memory-efficient attention."""
        if not self.has_xformers:
            raise RuntimeError("xFormers not available")
        
        # xFormers expects (B, S, H, D) format
        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        
        attn_bias = self.xops.LowerTriangularMask() if causal else None
        
        out = self.xops.memory_efficient_attention(
            Q_t, K_t, V_t,
            attn_bias=attn_bias
        )
        return out.transpose(1, 2)
    
    def benchmark_kernel(
        self,
        kernel_fn,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        warmup: int = 10,
        iterations: int = 100,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark an attention kernel."""
        # Warmup
        for _ in range(warmup):
            _ = kernel_fn(Q, K, V, **kwargs)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        
        for i in range(iterations):
            start_events[i].record()
            _ = kernel_fn(Q, K, V, **kwargs)
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
        batch_size: int = 8,
        num_heads: int = 12,
        head_dim: int = 64,
        dtype: str = 'float16',
        causal: bool = False
    ) -> List[Dict]:
        """Benchmark across sequence lengths."""
        if seq_lengths is None:
            seq_lengths = [512, 1024, 2048, 4096, 8192]
        
        results = []
        
        print(f"\nAttention Benchmark (B={batch_size}, H={num_heads}, D={head_dim})")
        print(f"Causal: {causal}, Dtype: {dtype}")
        print("=" * 70)
        
        for seq_len in seq_lengths:
            config = AttentionConfig(
                batch_size=batch_size,
                num_heads=num_heads,
                seq_length=seq_len,
                head_dim=head_dim,
                dtype=dtype,
                causal=causal
            )
            
            Q, K, V = self.setup(config)
            
            print(f"\nSeq Length: {seq_len}")
            print(f"  Standard memory: {config.memory_standard / 1e6:.1f} MB")
            print(f"  Flash memory:    {config.memory_flash / 1e6:.1f} MB")
            print("-" * 70)
            
            kernels = [
                ('standard', self.standard_attention),
            ]
            
            if self.has_flash:
                kernels.extend([
                    ('SDPA (auto)', self.sdpa_attention),
                ])
                # Try Flash-specific
                try:
                    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                        _ = F.scaled_dot_product_attention(Q, K, V)
                    kernels.append(('SDPA (flash)', self.sdpa_flash_only))
                except:
                    pass
                
                # Try memory-efficient
                try:
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                        _ = F.scaled_dot_product_attention(Q, K, V)
                    kernels.append(('SDPA (mem_eff)', self.sdpa_mem_efficient))
                except:
                    pass
            
            if self.has_xformers:
                kernels.append(('xFormers', self.xformers_attention))
            
            for name, kernel_fn in kernels:
                try:
                    result = self.benchmark_kernel(
                        kernel_fn, Q, K, V,
                        causal=causal
                    )
                    
                    tflops = config.flops / (result.mean_time_ms * 1e-3) / 1e12
                    
                    result_dict = {
                        'seq_length': seq_len,
                        'batch_size': batch_size,
                        'num_heads': num_heads,
                        'head_dim': head_dim,
                        'causal': causal,
                        'dtype': dtype,
                        'kernel': name,
                        'mean_ms': result.mean_time_ms,
                        'std_ms': result.std_time_ms,
                        'tflops': tflops,
                    }
                    results.append(result_dict)
                    
                    print(f"  {name:20s}: {result.mean_time_ms:8.3f} ms "
                          f"({tflops:6.2f} TFLOPS)")
                    
                except Exception as e:
                    print(f"  {name:20s}: SKIPPED - {e}")
            
            del Q, K, V
            torch.cuda.empty_cache()
        
        return results
    
    def run_memory_comparison(
        self,
        seq_length: int = 4096,
        batch_size: int = 8,
        num_heads: int = 12,
        head_dim: int = 64
    ) -> Dict:
        """Compare memory usage between implementations."""
        print(f"\nMemory Comparison @ seq={seq_length}")
        print("=" * 70)
        
        config = AttentionConfig(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_length=seq_length,
            head_dim=head_dim,
            dtype='float16'
        )
        
        results = {
            'config': {
                'batch_size': batch_size,
                'num_heads': num_heads,
                'seq_length': seq_length,
                'head_dim': head_dim,
            }
        }
        
        Q, K, V = self.setup(config)
        
        # Standard attention memory
        torch.cuda.reset_peak_memory_stats()
        _ = self.standard_attention(Q, K, V)
        torch.cuda.synchronize()
        standard_mem = torch.cuda.max_memory_allocated() / 1e6
        
        print(f"Standard attention: {standard_mem:.1f} MB peak")
        results['standard_MB'] = standard_mem
        
        # SDPA memory
        if self.has_flash:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = self.sdpa_attention(Q, K, V)
            torch.cuda.synchronize()
            sdpa_mem = torch.cuda.max_memory_allocated() / 1e6
            
            print(f"SDPA attention:     {sdpa_mem:.1f} MB peak")
            print(f"Memory savings:     {(1 - sdpa_mem/standard_mem)*100:.1f}%")
            results['sdpa_MB'] = sdpa_mem
            results['savings_percent'] = (1 - sdpa_mem/standard_mem) * 100
        
        return results
    
    def verify_correctness(
        self,
        config: AttentionConfig = None,
        rtol: float = 1e-3,
        atol: float = 1e-3
    ) -> bool:
        """Verify implementations produce same results."""
        if config is None:
            config = AttentionConfig(
                batch_size=2, num_heads=4, seq_length=128, head_dim=64,
                dtype='float32'
            )
        
        print(f"\nVerifying correctness")
        print("-" * 70)
        
        Q, K, V = self.setup(config)
        
        # Reference: standard attention
        ref = self.standard_attention(Q, K, V)
        
        tests = []
        if self.has_flash:
            tests.append(('SDPA', self.sdpa_attention(Q, K, V)))
        if self.has_xformers:
            tests.append(('xFormers', self.xformers_attention(Q, K, V)))
        
        all_passed = True
        for name, result in tests:
            max_diff = (result - ref).abs().max().item()
            passed = torch.allclose(result, ref, rtol=rtol, atol=atol)
            all_passed = all_passed and passed
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {name:20s}: {status} (max diff: {max_diff:.2e})")
        
        return all_passed
    
    def export_results(self, results: List[Dict], output_path: str):
        """Export results to JSON."""
        with open(output_path, 'w') as f:
            json.dump({'results': results}, f, indent=2)
        print(f"\nExported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Attention Benchmark')
    parser.add_argument('--seq-lengths', type=int, nargs='+',
                       default=[512, 1024, 2048, 4096])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-heads', type=int, default=12)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float32', 'float16'])
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--memory-test', action='store_true')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=str, default='attention_results.json')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    
    benchmark = AttentionBenchmark()
    
    if args.verify:
        benchmark.verify_correctness()
    
    results = benchmark.run_sequence_sweep(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=args.dtype,
        causal=args.causal
    )
    
    if args.memory_test:
        mem_results = benchmark.run_memory_comparison(
            seq_length=max(args.seq_lengths),
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            head_dim=args.head_dim
        )
        results.append({'memory_comparison': mem_results})
    
    benchmark.export_results(results, args.output)


if __name__ == "__main__":
    main()
