"""
easy_profiler.py - Zero-effort profiling utilities for ML training

Usage:
    from easy_profiler import profile_training, GPUMonitor, auto_profile

    # Option 1: Decorator (simplest)
    @auto_profile
    def train_epoch(model, dataloader):
        ...

    # Option 2: Context manager
    with profile_training("my_experiment"):
        train(model, dataloader)

    # Option 3: Background GPU monitoring
    with GPUMonitor(interval=1.0) as monitor:
        train(model, dataloader)
    print(monitor.summary())
"""

import torch
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from functools import wraps
from pathlib import Path
import json


@dataclass
class ProfilingResult:
    """Container for profiling results."""
    total_time_s: float = 0.0
    gpu_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    throughput_samples_per_sec: float = 0.0
    num_cuda_calls: int = 0
    bottleneck: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_s": self.total_time_s,
            "gpu_time_ms": self.gpu_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": self.avg_memory_mb,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "num_cuda_calls": self.num_cuda_calls,
            "bottleneck": self.bottleneck,
            "recommendations": self.recommendations
        }
    
    def __str__(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    PROFILING SUMMARY                         ║
╠══════════════════════════════════════════════════════════════╣
║  Total Time:        {self.total_time_s:>10.2f} s                        ║
║  GPU Time:          {self.gpu_time_ms:>10.2f} ms                       ║
║  CPU Time:          {self.cpu_time_ms:>10.2f} ms                       ║
║  Peak GPU Memory:   {self.peak_memory_mb:>10.2f} MB                       ║
║  Avg GPU Memory:    {self.avg_memory_mb:>10.2f} MB                       ║
║  Throughput:        {self.throughput_samples_per_sec:>10.2f} samples/s                ║
║  CUDA Calls:        {self.num_cuda_calls:>10d}                          ║
║  Bottleneck:        {self.bottleneck:<30}       ║
╚══════════════════════════════════════════════════════════════╝

Recommendations:
{chr(10).join(f"  • {r}" for r in self.recommendations) if self.recommendations else "  None"}
"""


class GPUMonitor:
    """
    Background GPU monitoring with minimal overhead.
    
    Usage:
        with GPUMonitor(interval=0.5) as monitor:
            train(model, dataloader)
        print(monitor.summary())
    """
    
    def __init__(self, interval: float = 1.0, device: int = 0):
        self.interval = interval
        self.device = device
        self.memory_samples: List[float] = []
        self.utilization_samples: List[float] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.start_time: float = 0
        self.end_time: float = 0
        
    def _monitor_loop(self):
        """Background monitoring thread."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            
            while not self._stop_event.is_set():
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.memory_samples.append(mem_info.used / 1024**2)  # MB
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.utilization_samples.append(util.gpu)
                
                self._stop_event.wait(self.interval)
                
            pynvml.nvmlShutdown()
        except ImportError:
            # Fallback to torch
            while not self._stop_event.is_set():
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated(self.device) / 1024**2
                    self.memory_samples.append(mem)
                self._stop_event.wait(self.interval)
    
    def __enter__(self):
        self.start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        return {
            "duration_s": self.end_time - self.start_time,
            "peak_memory_mb": max(self.memory_samples) if self.memory_samples else 0,
            "avg_memory_mb": sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            "min_memory_mb": min(self.memory_samples) if self.memory_samples else 0,
            "avg_utilization_pct": sum(self.utilization_samples) / len(self.utilization_samples) if self.utilization_samples else 0,
            "num_samples": len(self.memory_samples)
        }


@contextmanager
def profile_training(
    name: str = "training",
    output_dir: Optional[str] = None,
    trace_memory: bool = True,
    with_stack: bool = False,
    num_warmup_steps: int = 2,
    num_profile_steps: int = 5,
    record_shapes: bool = True,
):
    """
    Context manager for easy profiling of training loops.
    
    Usage:
        with profile_training("my_model") as profiler:
            for step, batch in enumerate(dataloader):
                loss = model(batch)
                loss.backward()
                optimizer.step()
                profiler.step()  # Important: call after each step
    
    Args:
        name: Name for the profiling run
        output_dir: Directory to save traces (default: ./profiling_output)
        trace_memory: Record memory allocation
        with_stack: Include Python call stacks (slower but more info)
        num_warmup_steps: Steps before profiling starts
        num_profile_steps: Steps to profile
        record_shapes: Record tensor shapes
    """
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
    
    output_dir = Path(output_dir or "./profiling_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    prof_schedule = schedule(
        wait=1,
        warmup=num_warmup_steps,
        active=num_profile_steps,
        repeat=1
    )
    
    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(str(output_dir / name)),
        record_shapes=record_shapes,
        profile_memory=trace_memory,
        with_stack=with_stack,
    ) as prof:
        yield prof
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Profiling Results: {name}")
    print(f"{'='*60}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    # Save detailed results
    results_file = output_dir / f"{name}_results.txt"
    with open(results_file, 'w') as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    
    print(f"\nResults saved to: {output_dir}")
    print(f"View in TensorBoard: tensorboard --logdir={output_dir}")


def auto_profile(func: Optional[Callable] = None, *, name: str = None, enabled: bool = True):
    """
    Decorator to automatically profile a function.
    
    Usage:
        @auto_profile
        def train_epoch(model, dataloader):
            for batch in dataloader:
                ...
        
        # Or with options
        @auto_profile(name="forward_pass", enabled=True)
        def forward(model, x):
            return model(x)
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not enabled or not torch.cuda.is_available():
                return fn(*args, **kwargs)
            
            profile_name = name or fn.__name__
            
            # Reset stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            start_time = time.time()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            result = fn(*args, **kwargs)
            end_event.record()
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            gpu_time_ms = start_event.elapsed_time(end_event)
            wall_time_s = end_time - start_time
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            
            print(f"\n[{profile_name}] Wall: {wall_time_s:.3f}s | GPU: {gpu_time_ms:.1f}ms | Peak Mem: {peak_memory_mb:.1f}MB")
            
            return result
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


class TrainingProfiler:
    """
    Drop-in profiler for training loops with automatic analysis.
    
    Usage:
        profiler = TrainingProfiler()
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                with profiler.profile_step():
                    loss = model(batch)
                    loss.backward()
                    optimizer.step()
        
        print(profiler.summary())
        profiler.save_report("training_profile.json")
    """
    
    def __init__(self, warmup_steps: int = 5):
        self.warmup_steps = warmup_steps
        self.step_times: List[float] = []
        self.gpu_times: List[float] = []
        self.memory_used: List[float] = []
        self.current_step = 0
        
    @contextmanager
    def profile_step(self, batch_size: int = 0):
        """Profile a single training step."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            yield
            return
        
        torch.cuda.synchronize()
        start_time = time.time()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        yield
        end_event.record()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        self.step_times.append(end_time - start_time)
        self.gpu_times.append(start_event.elapsed_time(end_event))
        self.memory_used.append(torch.cuda.memory_allocated() / 1024**2)
    
    def summary(self) -> ProfilingResult:
        """Generate profiling summary with recommendations."""
        if not self.step_times:
            return ProfilingResult()
        
        avg_step_time = sum(self.step_times) / len(self.step_times)
        avg_gpu_time = sum(self.gpu_times) / len(self.gpu_times)
        avg_cpu_time = (avg_step_time * 1000) - avg_gpu_time
        peak_memory = max(self.memory_used) if self.memory_used else 0
        avg_memory = sum(self.memory_used) / len(self.memory_used) if self.memory_used else 0
        
        # Determine bottleneck
        if avg_cpu_time > avg_gpu_time * 1.5:
            bottleneck = "CPU-bound (data loading?)"
        elif avg_gpu_time > avg_cpu_time * 2:
            bottleneck = "GPU compute"
        else:
            bottleneck = "Balanced"
        
        # Generate recommendations
        recommendations = []
        if avg_cpu_time > avg_gpu_time:
            recommendations.append("Consider increasing num_workers in DataLoader")
            recommendations.append("Enable pin_memory=True for faster CPU->GPU transfer")
        if peak_memory > 0.8 * torch.cuda.get_device_properties(0).total_memory / 1024**2:
            recommendations.append("Memory usage high - consider gradient checkpointing")
            recommendations.append("Try mixed precision training (AMP)")
        if avg_gpu_time < 10:  # Very fast steps
            recommendations.append("Steps are very fast - consider larger batch size")
        
        return ProfilingResult(
            total_time_s=sum(self.step_times),
            gpu_time_ms=avg_gpu_time,
            cpu_time_ms=avg_cpu_time,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            throughput_samples_per_sec=1.0 / avg_step_time if avg_step_time > 0 else 0,
            bottleneck=bottleneck,
            recommendations=recommendations
        )
    
    def save_report(self, path: str):
        """Save detailed profiling report."""
        report = {
            "summary": self.summary().to_dict(),
            "step_times": self.step_times,
            "gpu_times": self.gpu_times,
            "memory_samples": self.memory_used
        }
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)


# =============================================================================
# Quick profiling functions
# =============================================================================

def quick_benchmark(model: torch.nn.Module, input_shape: tuple, num_iterations: int = 100, warmup: int = 10) -> Dict[str, float]:
    """
    Quick benchmark for a model's forward pass.
    
    Usage:
        results = quick_benchmark(model, (1, 3, 224, 224))
        print(f"Forward pass: {results['avg_ms']:.2f} ms")
    """
    device = next(model.parameters()).device
    x = torch.randn(input_shape, device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(num_iterations):
        start_event.record()
        with torch.no_grad():
            _ = model(x)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
    
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        "throughput_per_sec": 1000 / (sum(times) / len(times))
    }


def memory_summary(device: int = 0) -> Dict[str, float]:
    """Get current GPU memory usage summary."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "cached_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_cached_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    import torch.nn as nn
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).cuda()
    
    print("=" * 60)
    print("Example 1: Quick Benchmark")
    print("=" * 60)
    results = quick_benchmark(model, (32, 1024))
    print(f"Forward pass: {results['avg_ms']:.2f} ± {results['std_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_per_sec']:.1f} batches/sec")
    
    print("\n" + "=" * 60)
    print("Example 2: TrainingProfiler")
    print("=" * 60)
    
    optimizer = torch.optim.Adam(model.parameters())
    profiler = TrainingProfiler(warmup_steps=2)
    
    for step in range(20):
        x = torch.randn(64, 1024, device='cuda')
        with profiler.profile_step():
            y = model(x)
            loss = y.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    print(profiler.summary())
    
    print("\n" + "=" * 60)
    print("Example 3: @auto_profile decorator")
    print("=" * 60)
    
    @auto_profile
    def forward_pass(model, x):
        return model(x)
    
    x = torch.randn(128, 1024, device='cuda')
    _ = forward_pass(model, x)
