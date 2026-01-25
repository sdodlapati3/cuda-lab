"""
Profiling utilities for GPU workloads.

This package provides tools for profiling single-GPU, multi-GPU, and multi-node
training workloads, including support for DeepSpeed and FSDP.

Quick Start:
    >>> from profiling_lab.utils import auto_profile, GPUMonitor, quick_benchmark
    
    # Decorator for automatic profiling
    >>> @auto_profile(warmup=3, iterations=10)
    ... def train_step(model, batch):
    ...     return model(batch)
    
    # Context manager for GPU monitoring
    >>> with GPUMonitor() as monitor:
    ...     for batch in dataloader:
    ...         train_step(batch)
    >>> print(monitor.get_summary())

Available modules:
    easy_profiler: Minimal-code profiling decorators and context managers
    unified_profiler: Comprehensive profiling for all GPU configurations
"""

from .easy_profiler import (
    auto_profile,
    GPUMonitor,
    TrainingProfiler,
    profile_training,
    quick_benchmark,
)

from .unified_profiler import (
    UnifiedProfiler,
    ProfilingConfig,
    ProfilerWrapper,
    NVTXAnnotator,
    get_profiler_context,
)

__all__ = [
    # easy_profiler
    'auto_profile',
    'GPUMonitor',
    'TrainingProfiler',
    'profile_training',
    'quick_benchmark',
    # unified_profiler
    'UnifiedProfiler',
    'ProfilingConfig',
    'ProfilerWrapper',
    'NVTXAnnotator',
    'get_profiler_context',
]
