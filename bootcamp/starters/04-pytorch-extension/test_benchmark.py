"""
Performance benchmarks for Fused LayerNorm

Compares:
1. PyTorch native layer_norm
2. Our fused CUDA implementation

Run with: python test_benchmark.py
"""

import torch
import torch.nn.functional as F
import time

try:
    from fused_layernorm import fused_layer_norm
    HAS_FUSED = True
except ImportError:
    HAS_FUSED = False
    print("Warning: fused_layernorm not built. Run 'pip install -e .' first")


def benchmark_fn(fn, warmup=10, iterations=100):
    """Benchmark a function using CUDA events for accurate timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations  # ms per iteration


def benchmark_layernorm():
    """Compare PyTorch vs Fused LayerNorm performance."""
    
    print("\n" + "="*70)
    print("FUSED LAYERNORM BENCHMARK")
    print("="*70)
    
    # Get device info
    props = torch.cuda.get_device_properties(0)
    print(f"\nDevice: {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print()
    
    # Test configurations (batch_size, hidden_size)
    configs = [
        (32, 768),      # BERT-base
        (32, 1024),     # GPT-2 small
        (32, 4096),     # GPT-2 large / LLaMA
        (64, 4096),     # Larger batch
        (128, 4096),    # Even larger
        (256, 4096),    # Production batch
        (32, 8192),     # Very wide
        (32, 16384),    # Extreme width
    ]
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║ Shape            │ PyTorch (μs) │ Fused (μs) │ Speedup │ Memory  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    
    for batch, hidden in configs:
        # Create inputs
        x = torch.randn(batch, hidden, device='cuda', requires_grad=True)
        w = torch.ones(hidden, device='cuda', requires_grad=True)
        b = torch.zeros(hidden, device='cuda', requires_grad=True)
        grad = torch.randn(batch, hidden, device='cuda')
        
        mem_mb = batch * hidden * 4 / (1024 * 1024)  # float32
        
        # Benchmark PyTorch
        def pytorch_fn():
            out = F.layer_norm(x, (hidden,), w, b, 1e-5)
            out.backward(grad)
        
        pytorch_time = benchmark_fn(pytorch_fn)
        
        # Benchmark Fused (if available)
        if HAS_FUSED:
            # Need fresh tensors to avoid grad accumulation
            x2 = torch.randn(batch, hidden, device='cuda', requires_grad=True)
            w2 = torch.ones(hidden, device='cuda', requires_grad=True)
            b2 = torch.zeros(hidden, device='cuda', requires_grad=True)
            
            def fused_fn():
                out = fused_layer_norm(x2, w2, b2, 1e-5)
                out.backward(grad)
            
            fused_time = benchmark_fn(fused_fn)
            speedup = pytorch_time / fused_time
        else:
            fused_time = float('nan')
            speedup = 0.0
        
        print(f"║ ({batch:3d}, {hidden:5d})    │ {pytorch_time*1000:11.1f} │ {fused_time*1000:10.1f} │ {speedup:6.2f}× │ {mem_mb:5.1f} MB ║")
    
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Forward-only benchmark
    print("\n--- Forward Pass Only ---\n")
    
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║ Shape            │ PyTorch (μs) │ Fused (μs) │ Speedup │ GB/s    ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    
    for batch, hidden in configs:
        x = torch.randn(batch, hidden, device='cuda')
        w = torch.ones(hidden, device='cuda')
        b = torch.zeros(hidden, device='cuda')
        
        # Forward only
        pytorch_time = benchmark_fn(lambda: F.layer_norm(x, (hidden,), w, b, 1e-5))
        
        if HAS_FUSED:
            fused_time = benchmark_fn(lambda: fused_layer_norm(x, w, b, 1e-5))
            speedup = pytorch_time / fused_time
        else:
            fused_time = float('nan')
            speedup = 0.0
        
        # Calculate bandwidth (read input + write output = 2 * size)
        bytes_accessed = 2 * batch * hidden * 4  # float32
        gb_per_s = (bytes_accessed / 1e9) / (fused_time / 1e3) if HAS_FUSED else 0
        
        print(f"║ ({batch:3d}, {hidden:5d})    │ {pytorch_time*1000:11.1f} │ {fused_time*1000:10.1f} │ {speedup:6.2f}× │ {gb_per_s:6.1f}  ║")
    
    print("╚══════════════════════════════════════════════════════════════════╝")


def profile_memory():
    """Profile memory usage."""
    print("\n--- Memory Usage ---\n")
    
    torch.cuda.reset_peak_memory_stats()
    
    batch, hidden = 128, 4096
    x = torch.randn(batch, hidden, device='cuda', requires_grad=True)
    w = torch.ones(hidden, device='cuda', requires_grad=True)
    b = torch.zeros(hidden, device='cuda', requires_grad=True)
    
    # PyTorch
    torch.cuda.reset_peak_memory_stats()
    out = F.layer_norm(x, (hidden,), w, b, 1e-5)
    out.sum().backward()
    pytorch_mem = torch.cuda.max_memory_allocated() / 1e6
    
    # Fused (if available)
    if HAS_FUSED:
        x2 = torch.randn(batch, hidden, device='cuda', requires_grad=True)
        w2 = torch.ones(hidden, device='cuda', requires_grad=True)
        b2 = torch.zeros(hidden, device='cuda', requires_grad=True)
        
        torch.cuda.reset_peak_memory_stats()
        out = fused_layer_norm(x2, w2, b2, 1e-5)
        out.sum().backward()
        fused_mem = torch.cuda.max_memory_allocated() / 1e6
        
        print(f"PyTorch peak memory: {pytorch_mem:.1f} MB")
        print(f"Fused peak memory:   {fused_mem:.1f} MB")
        print(f"Memory savings:      {(1 - fused_mem/pytorch_mem)*100:.1f}%")


if __name__ == '__main__':
    benchmark_layernorm()
    if HAS_FUSED:
        profile_memory()
    print()
