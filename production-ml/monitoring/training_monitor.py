#!/usr/bin/env python3
"""
training_monitor.py - Real-time monitoring for distributed training

Features:
- Metrics collection and aggregation across ranks
- Prometheus metrics export
- Integration with W&B/TensorBoard
- System metrics (GPU, CPU, memory)
- Throughput and efficiency tracking

Usage:
    from training_monitor import TrainingMonitor
    
    monitor = TrainingMonitor(project="my-project")
    
    for step in training_loop:
        loss = train_step()
        monitor.log({"loss": loss, "step": step})

Author: CUDA Lab
"""

import os
import sys
import time
import json
import atexit
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

import torch
import torch.distributed as dist


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MonitorConfig:
    """Configuration for training monitor."""
    project: str = "default"
    experiment: str = ""
    log_dir: str = "./logs"
    
    # Logging settings
    log_interval: int = 10  # Steps between logs
    log_to_console: bool = True
    log_to_file: bool = True
    
    # Prometheus settings
    enable_prometheus: bool = False
    prometheus_port: int = 8000
    
    # External loggers
    enable_wandb: bool = False
    enable_tensorboard: bool = True
    
    # System monitoring
    monitor_gpu: bool = True
    monitor_interval: float = 5.0  # Seconds between system metric collection


# ============================================================================
# Metrics Collection
# ============================================================================

class MetricsAggregator:
    """Aggregate metrics across distributed ranks."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.step_metrics: Dict[int, Dict[str, float]] = {}
    
    def add(self, name: str, value: float, step: Optional[int] = None):
        """Add a metric value."""
        self.metrics[name].append(value)
        if step is not None:
            if step not in self.step_metrics:
                self.step_metrics[step] = {}
            self.step_metrics[step][name] = value
    
    def get_average(self, name: str, last_n: Optional[int] = None) -> float:
        """Get average of metric."""
        values = self.metrics[name]
        if not values:
            return 0.0
        if last_n:
            values = values[-last_n:]
        return sum(values) / len(values)
    
    def get_last(self, name: str) -> float:
        """Get last value of metric."""
        values = self.metrics[name]
        return values[-1] if values else 0.0
    
    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.step_metrics.clear()


def reduce_metrics_across_ranks(
    metrics: Dict[str, float],
    device: torch.device,
) -> Dict[str, float]:
    """Reduce metrics across all ranks."""
    if not dist.is_initialized():
        return metrics
    
    reduced = {}
    for name, value in metrics.items():
        tensor = torch.tensor([value], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        reduced[name] = tensor.item()
    
    return reduced


# ============================================================================
# System Metrics
# ============================================================================

def get_gpu_metrics(device_id: int = 0) -> Dict[str, float]:
    """Get GPU metrics using PyTorch."""
    if not torch.cuda.is_available():
        return {}
    
    try:
        props = torch.cuda.get_device_properties(device_id)
        return {
            "gpu/memory_allocated_gb": torch.cuda.memory_allocated(device_id) / 1e9,
            "gpu/memory_reserved_gb": torch.cuda.memory_reserved(device_id) / 1e9,
            "gpu/memory_total_gb": props.total_memory / 1e9,
            "gpu/utilization": torch.cuda.utilization(device_id) if hasattr(torch.cuda, 'utilization') else 0,
        }
    except Exception as e:
        return {"gpu/error": str(e)}


def get_cpu_metrics() -> Dict[str, float]:
    """Get CPU and memory metrics."""
    try:
        import psutil
        return {
            "cpu/percent": psutil.cpu_percent(),
            "cpu/memory_percent": psutil.virtual_memory().percent,
            "cpu/memory_used_gb": psutil.virtual_memory().used / 1e9,
        }
    except ImportError:
        return {}


# ============================================================================
# Training Monitor
# ============================================================================

class TrainingMonitor:
    """
    Comprehensive training monitor for distributed training.
    
    Features:
    - Multi-rank metric aggregation
    - Console and file logging
    - Optional W&B and TensorBoard integration
    - System metrics collection
    """
    
    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        project: Optional[str] = None,
        experiment: Optional[str] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        self.config = config or MonitorConfig()
        
        if project:
            self.config.project = project
        if experiment:
            self.config.experiment = experiment
        
        # Get distributed info
        self.rank = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.world_size = world_size if world_size is not None else (dist.get_world_size() if dist.is_initialized() else 1)
        self.is_main = self.rank == 0
        
        # Metrics storage
        self.aggregator = MetricsAggregator()
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # Device
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        
        # Initialize logging
        self._setup_logging()
        
        # External loggers
        self._setup_external_loggers()
        
        # System monitoring thread
        self.system_metrics: Dict[str, float] = {}
        if self.config.monitor_gpu:
            self._start_system_monitor()
        
        # Register cleanup
        atexit.register(self.close)
    
    def _setup_logging(self):
        """Setup logging directories and handlers."""
        if not self.is_main:
            return
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = self.config.experiment or f"run_{timestamp}"
        self.log_dir = os.path.join(
            self.config.log_dir,
            self.config.project,
            self.experiment_name
        )
        os.makedirs(self.log_dir, exist_ok=True)
        
        # File logger
        if self.config.log_to_file:
            self.log_file = open(
                os.path.join(self.log_dir, "training.log"),
                "w"
            )
        else:
            self.log_file = None
    
    def _setup_external_loggers(self):
        """Setup W&B and TensorBoard."""
        self.wandb_run = None
        self.tb_writer = None
        
        if not self.is_main:
            return
        
        # Weights & Biases
        if self.config.enable_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.project,
                    name=self.experiment_name,
                )
            except ImportError:
                print("W&B not installed. Run: pip install wandb")
        
        # TensorBoard
        if self.config.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(
                    log_dir=os.path.join(self.log_dir, "tensorboard")
                )
            except ImportError:
                print("TensorBoard not installed. Run: pip install tensorboard")
    
    def _start_system_monitor(self):
        """Start background thread for system metrics."""
        self.monitor_running = True
        
        def monitor_loop():
            while self.monitor_running:
                try:
                    self.system_metrics.update(get_gpu_metrics(self.rank))
                    self.system_metrics.update(get_cpu_metrics())
                except Exception:
                    pass
                time.sleep(self.config.monitor_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (defaults to internal counter)
            commit: Whether to write to external loggers
        """
        if step is not None:
            self.step = step
        
        # Filter numeric metrics
        numeric_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))  # Filter NaN
        }
        
        # Add to aggregator
        for name, value in numeric_metrics.items():
            self.aggregator.add(name, value, self.step)
        
        # Add system metrics
        if self.system_metrics:
            numeric_metrics.update(self.system_metrics)
        
        # Add timing
        elapsed = time.time() - self.start_time
        numeric_metrics["time/elapsed_min"] = elapsed / 60
        numeric_metrics["time/step"] = self.step
        
        # Only log on main rank
        if not self.is_main:
            return
        
        # Check if should log
        if not commit or self.step % self.config.log_interval != 0:
            return
        
        # Console logging
        if self.config.log_to_console:
            self._log_to_console(numeric_metrics)
        
        # File logging
        if self.log_file:
            self._log_to_file(numeric_metrics)
        
        # External loggers
        if self.wandb_run:
            import wandb
            wandb.log(numeric_metrics, step=self.step)
        
        if self.tb_writer:
            for name, value in numeric_metrics.items():
                self.tb_writer.add_scalar(name, value, self.step)
        
        self.last_log_time = time.time()
    
    def _log_to_console(self, metrics: Dict[str, float]):
        """Format and print metrics to console."""
        # Format key metrics
        parts = [f"Step {self.step}"]
        
        # Priority metrics
        priority_keys = ["train/loss", "train/lr", "eval/loss", "eval/accuracy"]
        for key in priority_keys:
            if key in metrics:
                parts.append(f"{key.split('/')[-1]}={metrics[key]:.4f}")
        
        # Throughput
        if "train/samples_per_sec" in metrics:
            parts.append(f"throughput={metrics['train/samples_per_sec']:.0f}")
        
        # GPU memory
        if "gpu/memory_allocated_gb" in metrics:
            parts.append(f"mem={metrics['gpu/memory_allocated_gb']:.1f}GB")
        
        print(" | ".join(parts))
    
    def _log_to_file(self, metrics: Dict[str, float]):
        """Write metrics to JSON log file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.step,
            "epoch": self.epoch,
            "metrics": metrics,
        }
        self.log_file.write(json.dumps(log_entry) + "\n")
        self.log_file.flush()
    
    def set_epoch(self, epoch: int):
        """Set current epoch."""
        self.epoch = epoch
    
    def get_metric_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        summary = {}
        for name in self.aggregator.metrics:
            summary[f"{name}/avg"] = self.aggregator.get_average(name)
            summary[f"{name}/last"] = self.aggregator.get_last(name)
        return summary
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        if not self.is_main:
            return
        
        # Save to file
        hparams_file = os.path.join(self.log_dir, "hparams.json")
        with open(hparams_file, "w") as f:
            json.dump(hparams, f, indent=2)
        
        # Log to W&B
        if self.wandb_run:
            import wandb
            wandb.config.update(hparams)
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_hparams(
                hparams,
                {"dummy": 0},  # Required by TensorBoard
            )
    
    def close(self):
        """Cleanup resources."""
        self.monitor_running = False
        
        if self.is_main:
            if self.log_file:
                self.log_file.close()
            
            if self.tb_writer:
                self.tb_writer.close()
            
            if self.wandb_run:
                import wandb
                wandb.finish()


# ============================================================================
# Throughput Tracker
# ============================================================================

class ThroughputTracker:
    """Track training throughput."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.samples: List[int] = []
        self.times: List[float] = []
        self.start_time = None
    
    def start(self):
        """Start timing a batch."""
        self.start_time = time.perf_counter()
    
    def end(self, num_samples: int):
        """End timing and record samples processed."""
        if self.start_time is None:
            return
        
        elapsed = time.perf_counter() - self.start_time
        self.samples.append(num_samples)
        self.times.append(elapsed)
        
        # Keep window
        if len(self.samples) > self.window_size:
            self.samples = self.samples[-self.window_size:]
            self.times = self.times[-self.window_size:]
        
        self.start_time = None
    
    def get_throughput(self) -> float:
        """Get samples per second."""
        if not self.times:
            return 0.0
        return sum(self.samples) / sum(self.times)
    
    def get_batch_time_ms(self) -> float:
        """Get average batch time in milliseconds."""
        if not self.times:
            return 0.0
        return (sum(self.times) / len(self.times)) * 1000


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate training monitor."""
    print("=" * 60)
    print("TRAINING MONITOR DEMO")
    print("=" * 60)
    
    # Initialize
    config = MonitorConfig(
        project="demo",
        experiment="test_run",
        enable_tensorboard=True,
        log_interval=10,
    )
    
    monitor = TrainingMonitor(config=config)
    throughput = ThroughputTracker()
    
    # Log hyperparameters
    monitor.log_hyperparameters({
        "batch_size": 32,
        "learning_rate": 1e-4,
        "model": "transformer",
    })
    
    # Simulate training
    print("\nSimulating training...")
    
    for step in range(100):
        throughput.start()
        
        # Simulate training step
        time.sleep(0.01)
        loss = 1.0 / (step + 1) + 0.1 * (0.5 - torch.rand(1).item())
        
        throughput.end(num_samples=32)
        
        # Log metrics
        monitor.log({
            "train/loss": loss,
            "train/learning_rate": 1e-4,
            "train/samples_per_sec": throughput.get_throughput(),
            "train/batch_time_ms": throughput.get_batch_time_ms(),
        }, step=step)
    
    # Summary
    print("\nMetric Summary:")
    summary = monitor.get_metric_summary()
    for name, value in summary.items():
        if "loss" in name:
            print(f"  {name}: {value:.4f}")
    
    monitor.close()
    print("\nDemo complete! Check ./logs/demo/ for logs.")


if __name__ == "__main__":
    demo()
