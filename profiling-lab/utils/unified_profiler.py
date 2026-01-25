#!/usr/bin/env python
"""
unified_profiler.py - End-to-end profiling for single/multi-GPU/multi-node

Usage:
    # Single GPU
    python unified_profiler.py --mode single train.py
    
    # Multi-GPU (single node)
    python unified_profiler.py --mode multi-gpu --gpus 4 train.py
    
    # Multi-node (via SLURM)
    python unified_profiler.py --mode multi-node train.py

This script:
1. Detects the environment (single/multi-GPU/multi-node)
2. Sets up appropriate profiling (Nsight Systems + PyTorch Profiler)
3. Generates unified reports

Author: CUDA Lab
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import shutil


@dataclass
class ProfilingConfig:
    """Configuration for unified profiling."""
    mode: str = "auto"  # auto, single, multi-gpu, multi-node
    output_dir: str = "./profiling_results"
    use_nsys: bool = True
    use_pytorch_profiler: bool = True
    trace_cuda: bool = True
    trace_nvtx: bool = True
    trace_nccl: bool = True
    trace_memory: bool = True
    warmup_steps: int = 3
    profile_steps: int = 10
    gpus: Optional[int] = None


class UnifiedProfiler:
    """
    Unified profiler for all GPU configurations.
    
    Automatically detects environment and applies appropriate profiling.
    """
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect environment
        self.env_info = self._detect_environment()
        
        # Determine actual mode
        if config.mode == "auto":
            self.mode = self._auto_detect_mode()
        else:
            self.mode = config.mode
            
        print(f"\n{'='*60}")
        print(f"Unified Profiler - Mode: {self.mode.upper()}")
        print(f"{'='*60}")
        print(f"  GPUs detected: {self.env_info['num_gpus']}")
        print(f"  SLURM job: {self.env_info['is_slurm']}")
        print(f"  Nodes: {self.env_info['num_nodes']}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _detect_environment(self) -> Dict[str, Any]:
        """Detect the current environment."""
        import torch
        
        return {
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_available": torch.cuda.is_available(),
            "is_slurm": "SLURM_JOB_ID" in os.environ,
            "num_nodes": int(os.environ.get("SLURM_NNODES", 1)),
            "node_id": int(os.environ.get("SLURM_NODEID", 0)),
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        }
    
    def _auto_detect_mode(self) -> str:
        """Auto-detect the profiling mode."""
        if self.env_info["num_nodes"] > 1:
            return "multi-node"
        elif self.env_info["num_gpus"] > 1:
            return "multi-gpu"
        else:
            return "single"
    
    def _check_nsys_available(self) -> bool:
        """Check if Nsight Systems is available."""
        return shutil.which("nsys") is not None
    
    def _build_nsys_command(self, script_cmd: List[str]) -> List[str]:
        """Build Nsight Systems command."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{self.mode}_{timestamp}"
        
        if self.env_info["is_slurm"]:
            output_name += f"_node{self.env_info['node_id']}"
        
        nsys_cmd = [
            "nsys", "profile",
            "-o", str(self.output_dir / output_name),
            "--stats=true",
            "--force-overwrite=true",
        ]
        
        # Add trace options
        traces = []
        if self.config.trace_cuda:
            traces.append("cuda")
        if self.config.trace_nvtx:
            traces.append("nvtx")
        if self.config.trace_nccl and self.mode in ["multi-gpu", "multi-node"]:
            traces.append("nccl")  # Note: may need nvtx markers for NCCL
        
        traces.extend(["osrt", "cudnn", "cublas"])
        nsys_cmd.extend(["--trace=" + ",".join(traces)])
        
        # Add capture range for controlled profiling
        nsys_cmd.extend([
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
        ])
        
        nsys_cmd.extend(["--"])
        nsys_cmd.extend(script_cmd)
        
        return nsys_cmd
    
    def profile_script(self, script: str, script_args: List[str] = None):
        """Profile a Python script with appropriate tools."""
        script_args = script_args or []
        
        # Build the base command
        if self.mode == "single":
            base_cmd = ["python", script] + script_args
        elif self.mode == "multi-gpu":
            num_gpus = self.config.gpus or self.env_info["num_gpus"]
            base_cmd = [
                "torchrun",
                f"--nproc_per_node={num_gpus}",
                script
            ] + script_args
        else:  # multi-node
            base_cmd = [
                "python", "-m", "torch.distributed.run",
                f"--nnodes={self.env_info['num_nodes']}",
                f"--nproc_per_node={self.config.gpus or self.env_info['num_gpus']}",
                "--rdzv_backend=c10d",
                script
            ] + script_args
        
        # Wrap with Nsight Systems if available and requested
        if self.config.use_nsys and self._check_nsys_available():
            cmd = self._build_nsys_command(base_cmd)
            print(f"Running with Nsight Systems:")
            print(f"  {' '.join(cmd)}\n")
        else:
            cmd = base_cmd
            if self.config.use_nsys:
                print("Warning: Nsight Systems (nsys) not found, using PyTorch Profiler only")
        
        # Run the command
        env = os.environ.copy()
        env["UNIFIED_PROFILER_ENABLED"] = "1"
        env["UNIFIED_PROFILER_OUTPUT"] = str(self.output_dir)
        env["UNIFIED_PROFILER_WARMUP"] = str(self.config.warmup_steps)
        env["UNIFIED_PROFILER_STEPS"] = str(self.config.profile_steps)
        
        result = subprocess.run(cmd, env=env)
        
        return result.returncode
    
    def generate_report(self):
        """Generate a unified profiling report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode,
            "environment": self.env_info,
            "config": {
                "warmup_steps": self.config.warmup_steps,
                "profile_steps": self.config.profile_steps,
            },
            "files": []
        }
        
        # List generated files
        for f in self.output_dir.iterdir():
            if f.suffix in [".nsys-rep", ".json", ".txt"]:
                report["files"].append(str(f.name))
        
        # Save report
        report_path = self.output_dir / "profiling_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Profiling Complete!")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Files generated:")
        for f in report["files"]:
            print(f"  - {f}")
        print(f"\nTo view Nsight Systems results:")
        print(f"  nsys-ui {self.output_dir}/*.nsys-rep")
        print(f"\nTo view PyTorch Profiler results:")
        print(f"  tensorboard --logdir={self.output_dir}")
        print(f"{'='*60}\n")


# =============================================================================
# PyTorch Profiler Integration (to be used in training scripts)
# =============================================================================

def get_profiler_context():
    """
    Get a profiler context manager for use in training scripts.
    
    Usage in your training script:
        from unified_profiler import get_profiler_context
        
        profiler = get_profiler_context()
        
        for step, batch in enumerate(dataloader):
            with profiler:
                loss = train_step(batch)
            if profiler:
                profiler.step()
    """
    if os.environ.get("UNIFIED_PROFILER_ENABLED") != "1":
        return None
    
    import torch
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
    
    output_dir = os.environ.get("UNIFIED_PROFILER_OUTPUT", "./profiling_results")
    warmup = int(os.environ.get("UNIFIED_PROFILER_WARMUP", 3))
    active = int(os.environ.get("UNIFIED_PROFILER_STEPS", 10))
    
    # Get rank info for distributed
    rank = int(os.environ.get("RANK", 0))
    
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=1,
            warmup=warmup,
            active=active,
            repeat=1
        ),
        on_trace_ready=tensorboard_trace_handler(
            f"{output_dir}/pytorch_profiler_rank{rank}"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


class ProfilerWrapper:
    """
    Easy wrapper for profiling training loops.
    
    Usage:
        profiler = ProfilerWrapper()
        
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                with profiler.profile_step(step):
                    loss = train_step(batch)
        
        profiler.finish()
    """
    
    def __init__(self, enabled: bool = None):
        import torch
        
        if enabled is None:
            enabled = os.environ.get("UNIFIED_PROFILER_ENABLED") == "1"
        
        self.enabled = enabled and torch.cuda.is_available()
        self.profiler = None
        self.step_count = 0
        
        if self.enabled:
            self._setup_profiler()
    
    def _setup_profiler(self):
        """Set up the PyTorch profiler."""
        import torch
        from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
        
        output_dir = os.environ.get("UNIFIED_PROFILER_OUTPUT", "./profiling_results")
        warmup = int(os.environ.get("UNIFIED_PROFILER_WARMUP", 3))
        active = int(os.environ.get("UNIFIED_PROFILER_STEPS", 10))
        rank = int(os.environ.get("RANK", 0))
        
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=warmup, active=active, repeat=1),
            on_trace_ready=tensorboard_trace_handler(f"{output_dir}/pytorch_rank{rank}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self.profiler.__enter__()
        
        # Enable CUDA profiler API for Nsight Systems capture range
        torch.cuda.cudart().cudaProfilerStart()
    
    def profile_step(self, step: int = None):
        """Context manager for profiling a single step."""
        import torch
        
        class StepContext:
            def __init__(self, wrapper):
                self.wrapper = wrapper
            
            def __enter__(self):
                if self.wrapper.enabled:
                    torch.cuda.nvtx.range_push(f"step_{self.wrapper.step_count}")
                return self
            
            def __exit__(self, *args):
                if self.wrapper.enabled:
                    torch.cuda.nvtx.range_pop()
                    if self.wrapper.profiler:
                        self.wrapper.profiler.step()
                    self.wrapper.step_count += 1
        
        return StepContext(self)
    
    def finish(self):
        """Finish profiling and generate reports."""
        import torch
        
        if self.enabled:
            torch.cuda.cudart().cudaProfilerStop()
            if self.profiler:
                self.profiler.__exit__(None, None, None)
                
                # Print summary
                print("\n" + "="*60)
                print("PyTorch Profiler Summary")
                print("="*60)
                print(self.profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=20
                ))


# =============================================================================
# NVTX Markers for Custom Annotations
# =============================================================================

class NVTXAnnotator:
    """
    NVTX annotation helper for marking code sections.
    
    Usage:
        annotator = NVTXAnnotator()
        
        with annotator.range("data_loading"):
            batch = next(dataloader)
        
        with annotator.range("forward"):
            output = model(batch)
        
        with annotator.range("backward"):
            loss.backward()
    """
    
    def __init__(self, enabled: bool = True):
        import torch
        self.enabled = enabled and torch.cuda.is_available()
        self.nvtx = torch.cuda.nvtx if self.enabled else None
    
    def range(self, name: str, color: str = None):
        """Create an NVTX range context manager."""
        class NVTXRange:
            def __init__(self, annotator, name):
                self.annotator = annotator
                self.name = name
            
            def __enter__(self):
                if self.annotator.enabled:
                    self.annotator.nvtx.range_push(self.name)
                return self
            
            def __exit__(self, *args):
                if self.annotator.enabled:
                    self.annotator.nvtx.range_pop()
        
        return NVTXRange(self, name)
    
    def mark(self, name: str):
        """Create an NVTX marker (instant event)."""
        if self.enabled:
            self.nvtx.mark(name)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified GPU profiling for single/multi-GPU/multi-node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile single GPU training
  python unified_profiler.py train.py --epochs 10
  
  # Profile multi-GPU training
  python unified_profiler.py --mode multi-gpu --gpus 4 train.py
  
  # Profile with only PyTorch Profiler (no nsys)
  python unified_profiler.py --no-nsys train.py
  
  # Custom output directory
  python unified_profiler.py -o ./my_profiles train.py
        """
    )
    
    parser.add_argument("script", help="Python script to profile")
    parser.add_argument("script_args", nargs="*", help="Arguments for the script")
    parser.add_argument("--mode", choices=["auto", "single", "multi-gpu", "multi-node"],
                       default="auto", help="Profiling mode")
    parser.add_argument("--gpus", type=int, help="Number of GPUs (for multi-GPU mode)")
    parser.add_argument("-o", "--output", default="./profiling_results",
                       help="Output directory")
    parser.add_argument("--no-nsys", action="store_true",
                       help="Disable Nsight Systems profiling")
    parser.add_argument("--no-pytorch", action="store_true",
                       help="Disable PyTorch Profiler")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Warmup steps before profiling")
    parser.add_argument("--steps", type=int, default=10,
                       help="Number of steps to profile")
    
    args = parser.parse_args()
    
    config = ProfilingConfig(
        mode=args.mode,
        output_dir=args.output,
        use_nsys=not args.no_nsys,
        use_pytorch_profiler=not args.no_pytorch,
        warmup_steps=args.warmup,
        profile_steps=args.steps,
        gpus=args.gpus,
    )
    
    profiler = UnifiedProfiler(config)
    
    return_code = profiler.profile_script(args.script, args.script_args)
    profiler.generate_report()
    
    sys.exit(return_code)


if __name__ == "__main__":
    main()
