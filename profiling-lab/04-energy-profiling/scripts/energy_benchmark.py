"""
energy_benchmark.py - GPU Energy Profiling Utilities

Measure power consumption and energy efficiency of GPU workloads.

Requirements:
    pip install pynvml

Usage:
    from energy_benchmark import EnergyBenchmark, measure_energy
    
    # As context manager
    with measure_energy() as result:
        run_my_workload()
    print(f"Energy: {result.energy_joules:.2f} J")
    
    # As decorator
    @measure_energy_decorator
    def my_function():
        ...
    
    # Full benchmark
    bench = EnergyBenchmark()
    for impl_name, impl_fn in implementations.items():
        bench.measure(impl_name, impl_fn)
    bench.print_results()
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List
from contextlib import contextmanager

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install pynvml")


@dataclass
class EnergyResult:
    """Results from energy measurement."""
    name: str = ""
    duration_s: float = 0.0
    energy_joules: float = 0.0
    avg_power_watts: float = 0.0
    peak_power_watts: float = 0.0
    min_power_watts: float = 0.0
    samples: int = 0
    
    @property
    def efficiency_flops_per_joule(self) -> float:
        """Calculate efficiency if FLOPS are known."""
        return 0.0  # Override if FLOPS measured
    
    def __str__(self):
        return (
            f"{self.name}: {self.duration_s:.2f}s, "
            f"{self.energy_joules:.1f}J, "
            f"{self.avg_power_watts:.0f}W avg "
            f"({self.min_power_watts:.0f}-{self.peak_power_watts:.0f}W)"
        )


class PowerMonitor:
    """Background thread for continuous power monitoring."""
    
    def __init__(self, device_index: int = 0, sample_interval_ms: int = 10):
        """
        Args:
            device_index: GPU index to monitor
            sample_interval_ms: Sampling interval in milliseconds
        """
        self.device_index = device_index
        self.sample_interval = sample_interval_ms / 1000.0
        self.power_samples: List[float] = []
        self.timestamps: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        
        if NVML_AVAILABLE:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    
    def _sample_loop(self):
        """Sampling thread main loop."""
        while self._running:
            if self._handle:
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                    power_w = power_mw / 1000.0
                    self.power_samples.append(power_w)
                    self.timestamps.append(time.perf_counter())
                except pynvml.NVMLError:
                    pass
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start power monitoring."""
        self.power_samples = []
        self.timestamps = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> EnergyResult:
        """Stop monitoring and return results."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.power_samples:
            return EnergyResult()
        
        # Calculate metrics
        duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
        avg_power = sum(self.power_samples) / len(self.power_samples)
        peak_power = max(self.power_samples)
        min_power = min(self.power_samples)
        
        # Integrate power over time for energy (trapezoidal rule)
        energy = 0.0
        for i in range(1, len(self.timestamps)):
            dt = self.timestamps[i] - self.timestamps[i-1]
            avg_p = (self.power_samples[i] + self.power_samples[i-1]) / 2
            energy += avg_p * dt
        
        return EnergyResult(
            duration_s=duration,
            energy_joules=energy,
            avg_power_watts=avg_power,
            peak_power_watts=peak_power,
            min_power_watts=min_power,
            samples=len(self.power_samples),
        )


@contextmanager
def measure_energy(device_index: int = 0, sample_interval_ms: int = 10):
    """
    Context manager for measuring energy consumption.
    
    Usage:
        with measure_energy() as result:
            run_workload()
        print(f"Energy: {result.energy_joules:.2f} J")
    """
    monitor = PowerMonitor(device_index, sample_interval_ms)
    result = EnergyResult()
    
    try:
        monitor.start()
        yield result
    finally:
        measured = monitor.stop()
        # Copy results to yielded object
        result.duration_s = measured.duration_s
        result.energy_joules = measured.energy_joules
        result.avg_power_watts = measured.avg_power_watts
        result.peak_power_watts = measured.peak_power_watts
        result.min_power_watts = measured.min_power_watts
        result.samples = measured.samples


def measure_energy_decorator(func: Callable) -> Callable:
    """Decorator to measure energy of a function."""
    def wrapper(*args, **kwargs):
        with measure_energy() as result:
            ret = func(*args, **kwargs)
        print(f"{func.__name__}: {result}")
        return ret
    return wrapper


class EnergyBenchmark:
    """
    Comprehensive energy benchmarking utility.
    
    Usage:
        bench = EnergyBenchmark()
        
        def workload_v1():
            ...
        
        def workload_v2():
            ...
        
        bench.measure("v1", workload_v1, iterations=10)
        bench.measure("v2", workload_v2, iterations=10)
        bench.print_results()
        bench.save_results("energy_results.json")
    """
    
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.results: Dict[str, EnergyResult] = {}
        
        # Get device info
        if NVML_AVAILABLE:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.device_name = pynvml.nvmlDeviceGetName(handle)
            self.tdp = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
        else:
            self.device_name = "Unknown"
            self.tdp = 400
    
    def measure(
        self,
        name: str,
        workload: Callable,
        iterations: int = 1,
        warmup: int = 1,
    ) -> EnergyResult:
        """
        Measure energy consumption of a workload.
        
        Args:
            name: Name for this measurement
            workload: Function to measure
            iterations: Number of iterations to run
            warmup: Warmup iterations (not measured)
        
        Returns:
            EnergyResult with measurements
        """
        # Warmup
        for _ in range(warmup):
            workload()
        
        # Synchronize GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        # Measure
        with measure_energy(self.device_index) as result:
            for _ in range(iterations):
                workload()
            
            # Final sync
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass
        
        result.name = name
        
        # Normalize by iterations
        if iterations > 1:
            result.duration_s /= iterations
            result.energy_joules /= iterations
        
        self.results[name] = result
        return result
    
    def print_results(self):
        """Print benchmark results as table."""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{'='*70}")
        print(f"Energy Benchmark Results")
        print(f"Device: {self.device_name} (TDP: {self.tdp:.0f}W)")
        print(f"{'='*70}")
        
        print(f"{'Name':<20} {'Time (s)':<12} {'Energy (J)':<12} {'Avg Power':<12} {'Peak Power':<12}")
        print("-" * 70)
        
        # Sort by energy
        for name, result in sorted(self.results.items(), key=lambda x: x[1].energy_joules):
            print(f"{name:<20} {result.duration_s:<12.3f} {result.energy_joules:<12.1f} "
                  f"{result.avg_power_watts:<12.0f} {result.peak_power_watts:<12.0f}")
        
        print("=" * 70)
        
        # Efficiency comparison
        if len(self.results) > 1:
            baseline = list(self.results.values())[0]
            print("\nRelative efficiency (vs first):")
            for name, result in self.results.items():
                if result.energy_joules > 0:
                    efficiency = baseline.energy_joules / result.energy_joules
                    print(f"  {name}: {efficiency:.2f}x")
    
    def save_results(self, path: str):
        """Save results to JSON file."""
        import json
        
        data = {
            'device': self.device_name,
            'tdp_watts': self.tdp,
            'results': {
                name: {
                    'duration_s': r.duration_s,
                    'energy_joules': r.energy_joules,
                    'avg_power_watts': r.avg_power_watts,
                    'peak_power_watts': r.peak_power_watts,
                    'samples': r.samples,
                }
                for name, r in self.results.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {path}")


# Quick power reading function
def get_current_power(device_index: int = 0) -> float:
    """Get current power draw in watts."""
    if not NVML_AVAILABLE:
        return 0.0
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
    return power_mw / 1000.0


# Example usage
if __name__ == "__main__":
    import torch
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        exit(1)
    
    print(f"Current power: {get_current_power():.0f}W")
    
    # Create benchmark
    bench = EnergyBenchmark()
    
    # Define workloads
    N = 4096
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')
    
    def matmul_fp32():
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
    
    def matmul_fp16():
        with torch.amp.autocast('cuda'):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
    
    # Measure
    bench.measure("matmul_fp32", matmul_fp32, iterations=10, warmup=2)
    bench.measure("matmul_fp16", matmul_fp16, iterations=10, warmup=2)
    
    # Results
    bench.print_results()
    bench.save_results("energy_results.json")
