"""
compile_benchmark.py - PyTorch 2.0 torch.compile Benchmarks

Compare eager mode vs compiled mode across different:
- Model architectures
- Compilation modes
- Batch sizes

Author: CUDA Lab
"""

import torch
import torch.nn as nn
import time
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    mode: str
    mean_ms: float
    std_ms: float
    speedup: float = 1.0


class SimpleMLP(nn.Module):
    """Simple MLP for compile testing."""
    
    def __init__(self, input_dim=768, hidden_dim=3072, output_dim=768, n_layers=4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    """Single transformer block for compile testing."""
    
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # FFN with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class MiniTransformer(nn.Module):
    """Small transformer for compile testing."""
    
    def __init__(self, d_model=768, n_heads=12, n_layers=6, d_ff=3072):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ConvNet(nn.Module):
    """Simple CNN for compile testing."""
    
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    warmup: int = 50,
    iterations: int = 100,
    device: torch.device = None
) -> Tuple[float, float]:
    """
    Benchmark a model's forward pass.
    
    Returns:
        (mean_time_ms, std_time_ms)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_shape, device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        with torch.no_grad():
            _ = model(x)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return sum(times) / len(times), torch.tensor(times).std().item()


def compare_compile_modes(
    model_fn,
    model_name: str,
    input_shape: Tuple[int, ...],
    device: torch.device
) -> Dict[str, BenchmarkResult]:
    """Compare different compilation modes."""
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Input shape: {input_shape}")
    print(f"{'='*60}")
    
    results = {}
    
    # Test modes
    modes = [
        ('eager', None),
        ('default', 'default'),
        ('reduce-overhead', 'reduce-overhead'),
        ('max-autotune', 'max-autotune'),
    ]
    
    eager_time = None
    
    for mode_name, compile_mode in modes:
        print(f"\nTesting: {mode_name}")
        
        # Create fresh model
        model = model_fn()
        
        # Compile if not eager
        if compile_mode is not None:
            try:
                model = torch.compile(model, mode=compile_mode)
            except Exception as e:
                print(f"  Compilation failed: {e}")
                continue
        
        try:
            mean_ms, std_ms = benchmark_model(model, input_shape, device=device)
            
            # Calculate speedup vs eager
            if eager_time is None:
                eager_time = mean_ms
                speedup = 1.0
            else:
                speedup = eager_time / mean_ms
            
            results[mode_name] = BenchmarkResult(
                mode=mode_name,
                mean_ms=mean_ms,
                std_ms=std_ms,
                speedup=speedup
            )
            
            print(f"  Time: {mean_ms:.3f} ± {std_ms:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  Benchmark failed: {e}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return results


def benchmark_compilation_overhead(
    model_fn,
    input_shape: Tuple[int, ...],
    device: torch.device
):
    """Measure compilation time overhead."""
    
    print("\n" + "="*60)
    print("Compilation Overhead")
    print("="*60)
    
    model = model_fn()
    x = torch.randn(*input_shape, device=device)
    
    # Time compilation
    start = time.perf_counter()
    compiled_model = torch.compile(model, mode='default')
    compile_time = time.perf_counter() - start
    
    print(f"\nCompilation time: {compile_time*1000:.1f} ms")
    
    # First inference (includes graph capture)
    compiled_model = compiled_model.to(device)
    
    start = time.perf_counter()
    with torch.no_grad():
        _ = compiled_model(x)
    torch.cuda.synchronize()
    first_run = time.perf_counter() - start
    
    print(f"First inference (graph capture): {first_run*1000:.1f} ms")
    
    # Subsequent inference
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = compiled_model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    mean_time = sum(times) / len(times)
    print(f"Subsequent inference: {mean_time*1000:.3f} ms")
    
    # Break-even point
    break_even = int(first_run / (mean_time * 0.3))  # Assuming 30% speedup
    print(f"\nBreak-even point: ~{break_even} iterations")


def benchmark_dynamic_shapes(device: torch.device):
    """Test torch.compile with dynamic shapes."""
    
    print("\n" + "="*60)
    print("Dynamic Shapes Benchmark")
    print("="*60)
    
    model = SimpleMLP()
    
    # Without dynamic shapes (recompiles for each shape)
    print("\nStatic shapes (separate compilations):")
    model_static = torch.compile(model, mode='default', dynamic=False)
    model_static = model_static.to(device)
    
    for batch_size in [32, 64, 128]:
        x = torch.randn(batch_size, 768, device=device)
        
        # Warmup/compile
        for _ in range(10):
            with torch.no_grad():
                _ = model_static(x)
        
        mean_ms, std_ms = benchmark_model(model_static, (batch_size, 768), 
                                          warmup=10, iterations=50, device=device)
        print(f"  Batch {batch_size}: {mean_ms:.3f} ms")
    
    # With dynamic shapes
    print("\nDynamic shapes (single compilation):")
    model = SimpleMLP()
    model_dynamic = torch.compile(model, mode='default', dynamic=True)
    model_dynamic = model_dynamic.to(device)
    
    # Single warmup
    x = torch.randn(32, 768, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model_dynamic(x)
    
    for batch_size in [32, 64, 128]:
        x = torch.randn(batch_size, 768, device=device)
        
        mean_ms, std_ms = benchmark_model(model_dynamic, (batch_size, 768),
                                          warmup=5, iterations=50, device=device)
        print(f"  Batch {batch_size}: {mean_ms:.3f} ms")


def run_full_benchmark(device: torch.device):
    """Run comprehensive torch.compile benchmarks."""
    
    all_results = {}
    
    # MLP
    results = compare_compile_modes(
        lambda: SimpleMLP(),
        "SimpleMLP",
        (64, 768),
        device
    )
    all_results['mlp'] = {k: v.__dict__ for k, v in results.items()}
    
    # Transformer
    results = compare_compile_modes(
        lambda: MiniTransformer(n_layers=4),
        "MiniTransformer (4 layers)",
        (8, 256, 768),
        device
    )
    all_results['transformer'] = {k: v.__dict__ for k, v in results.items()}
    
    # ConvNet
    results = compare_compile_modes(
        lambda: ConvNet(),
        "ConvNet",
        (32, 3, 224, 224),
        device
    )
    all_results['convnet'] = {k: v.__dict__ for k, v in results.items()}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        for mode, result in results.items():
            print(f"  {mode:20s}: {result['mean_ms']:7.3f} ms ({result['speedup']:.2f}x)")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='torch.compile Benchmark')
    parser.add_argument('--full', action='store_true', help='Run full benchmark')
    parser.add_argument('--overhead', action='store_true', 
                       help='Measure compilation overhead')
    parser.add_argument('--dynamic', action='store_true',
                       help='Test dynamic shapes')
    parser.add_argument('--output', type=str, default=None,
                       help='Save results to JSON')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA required for meaningful benchmarks")
        return
    
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    
    # Check torch.compile availability
    if not hasattr(torch, 'compile'):
        print("torch.compile not available (requires PyTorch 2.0+)")
        return
    
    if args.overhead:
        benchmark_compilation_overhead(
            lambda: SimpleMLP(),
            (64, 768),
            device
        )
    
    if args.dynamic:
        benchmark_dynamic_shapes(device)
    
    if args.full or (not args.overhead and not args.dynamic):
        results = run_full_benchmark(device)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
