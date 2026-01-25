#!/usr/bin/env python3
"""
experiment_tracking.py - MLOps experiment tracking and management

Features:
- Experiment versioning and tracking
- Hyperparameter logging
- Metric tracking with history
- Artifact management
- Integration with W&B, MLflow, TensorBoard
- Reproducibility (save code, config, environment)

Usage:
    tracker = ExperimentTracker(
        project="my_project",
        experiment_name="baseline_v1",
    )
    
    tracker.log_params({"lr": 1e-4, "batch_size": 32})
    tracker.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
    tracker.log_artifact("model.pt")
    
    tracker.finish()

Author: CUDA Lab
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

# Try to import optional dependencies
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    # Basic info
    project: str = "default"
    experiment_name: str = "experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Storage
    base_dir: str = "./experiments"
    
    # Backends
    use_wandb: bool = False
    use_mlflow: bool = False
    use_tensorboard: bool = True
    
    # W&B settings
    wandb_entity: Optional[str] = None
    
    # MLflow settings
    mlflow_tracking_uri: str = "file:./mlruns"
    
    # Reproducibility
    save_code: bool = True
    save_env: bool = True


class ExperimentTracker:
    """
    Unified experiment tracking with multiple backend support.
    
    Provides:
    - Consistent API across backends
    - Automatic versioning
    - Reproducibility features
    - Artifact management
    """
    
    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        project: str = "default",
        experiment_name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        use_wandb: bool = False,
        use_mlflow: bool = False,
        use_tensorboard: bool = True,
    ):
        self.config = config or ExperimentConfig()
        
        # Override config with explicit args
        self.config.project = project
        self.config.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config.description = description
        self.config.tags = tags or []
        self.config.use_wandb = use_wandb and HAS_WANDB
        self.config.use_mlflow = use_mlflow and HAS_MLFLOW
        self.config.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        
        # Create experiment directory
        self.exp_dir = Path(self.config.base_dir) / self.config.project / self.config.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for local tracking
        self.params: Dict[str, Any] = {}
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.artifacts: List[str] = []
        
        # Initialize backends
        self._init_backends()
        
        # Save reproducibility info
        if self.config.save_code:
            self._save_code_snapshot()
        if self.config.save_env:
            self._save_environment()
        
        logger.info(f"Experiment initialized: {self.config.experiment_name}")
        logger.info(f"Directory: {self.exp_dir}")
    
    def _init_backends(self):
        """Initialize tracking backends."""
        # W&B
        if self.config.use_wandb:
            wandb.init(
                project=self.config.project,
                name=self.config.experiment_name,
                entity=self.config.wandb_entity,
                tags=self.config.tags,
                notes=self.config.description,
                dir=str(self.exp_dir),
            )
            logger.info("W&B initialized")
        
        # MLflow
        if self.config.use_mlflow:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.project)
            mlflow.start_run(run_name=self.config.experiment_name)
            if self.config.tags:
                mlflow.set_tags({f"tag_{i}": t for i, t in enumerate(self.config.tags)})
            logger.info("MLflow initialized")
        
        # TensorBoard
        if self.config.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.exp_dir / "tensorboard"))
            logger.info("TensorBoard initialized")
        else:
            self.tb_writer = None
    
    def _save_code_snapshot(self):
        """Save snapshot of current code."""
        code_dir = self.exp_dir / "code"
        code_dir.mkdir(exist_ok=True)
        
        # Try to get git info
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            git_diff = subprocess.check_output(
                ["git", "diff"], stderr=subprocess.DEVNULL
            ).decode()
            
            with open(code_dir / "git_info.txt", "w") as f:
                f.write(f"Commit: {git_hash}\n\n")
                f.write("Uncommitted changes:\n")
                f.write(git_diff)
            
            logger.info(f"Git commit: {git_hash}")
        except:
            logger.warning("Could not get git info")
        
        # Save main script
        if hasattr(sys, 'argv') and sys.argv:
            main_script = sys.argv[0]
            if os.path.exists(main_script):
                shutil.copy(main_script, code_dir / "main_script.py")
    
    def _save_environment(self):
        """Save environment information."""
        env_file = self.exp_dir / "environment.json"
        
        env_info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Get installed packages
        try:
            result = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
            env_info["packages"] = result.decode().strip().split("\n")
        except:
            pass
        
        # Get CUDA info
        try:
            import torch
            env_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env_info["cuda_version"] = torch.version.cuda
                env_info["cudnn_version"] = str(torch.backends.cudnn.version())
                env_info["gpu_count"] = torch.cuda.device_count()
                env_info["gpu_names"] = [
                    torch.cuda.get_device_name(i) 
                    for i in range(torch.cuda.device_count())
                ]
        except:
            pass
        
        with open(env_file, "w") as f:
            json.dump(env_info, f, indent=2)
    
    # ========================================================================
    # Logging API
    # ========================================================================
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self.params.update(params)
        
        # Save locally
        params_file = self.exp_dir / "params.json"
        with open(params_file, "w") as f:
            json.dump(self.params, f, indent=2)
        
        # Log to backends
        if self.config.use_wandb:
            wandb.config.update(params)
        
        if self.config.use_mlflow:
            mlflow.log_params(params)
        
        if self.tb_writer:
            # TensorBoard doesn't have native param logging, use text
            param_text = "\n".join(f"{k}: {v}" for k, v in params.items())
            self.tb_writer.add_text("hyperparameters", param_text)
        
        logger.info(f"Logged params: {list(params.keys())}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """Log metrics at a step."""
        # Store in history
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append({
                "value": value,
                "step": step,
                "timestamp": datetime.now().isoformat(),
            })
        
        # Log to backends
        if self.config.use_wandb:
            wandb.log(metrics, step=step, commit=commit)
        
        if self.config.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        
        if self.tb_writer and step is not None:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ):
        """Log a single metric."""
        self.log_metrics({name: value}, step=step)
    
    def log_artifact(
        self,
        local_path: str,
        artifact_name: Optional[str] = None,
        artifact_type: str = "file",
    ):
        """Log an artifact (file or directory)."""
        local_path = Path(local_path)
        artifact_name = artifact_name or local_path.name
        
        # Copy to experiment directory
        artifact_dir = self.exp_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        
        dest_path = artifact_dir / artifact_name
        if local_path.is_dir():
            shutil.copytree(local_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy(local_path, dest_path)
        
        self.artifacts.append(str(dest_path))
        
        # Log to backends
        if self.config.use_wandb:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            if local_path.is_dir():
                artifact.add_dir(str(local_path))
            else:
                artifact.add_file(str(local_path))
            wandb.log_artifact(artifact)
        
        if self.config.use_mlflow:
            mlflow.log_artifact(str(local_path))
        
        logger.info(f"Logged artifact: {artifact_name}")
    
    def log_model(
        self,
        model,
        model_name: str = "model",
        save_format: str = "torch",  # "torch", "onnx", "torchscript"
    ):
        """Log a model checkpoint."""
        import torch
        
        model_dir = self.exp_dir / "models"
        model_dir.mkdir(exist_ok=True)
        
        if save_format == "torch":
            model_path = model_dir / f"{model_name}.pt"
            torch.save(model.state_dict(), model_path)
        elif save_format == "torchscript":
            model_path = model_dir / f"{model_name}.ts"
            scripted = torch.jit.script(model)
            scripted.save(str(model_path))
        elif save_format == "onnx":
            model_path = model_dir / f"{model_name}.onnx"
            # Requires dummy input - simplified example
            raise NotImplementedError("ONNX export requires dummy input")
        
        self.artifacts.append(str(model_path))
        
        # Log to backends
        if self.config.use_wandb:
            artifact = wandb.Artifact(model_name, type="model")
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
        
        if self.config.use_mlflow:
            mlflow.log_artifact(str(model_path))
        
        logger.info(f"Logged model: {model_name}")
    
    def log_figure(
        self,
        figure,
        name: str,
        step: Optional[int] = None,
    ):
        """Log a matplotlib figure."""
        import matplotlib.pyplot as plt
        
        fig_dir = self.exp_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        
        fig_path = fig_dir / f"{name}.png"
        figure.savefig(fig_path, dpi=150, bbox_inches="tight")
        
        # Log to backends
        if self.config.use_wandb:
            wandb.log({name: wandb.Image(str(fig_path))}, step=step)
        
        if self.tb_writer and step is not None:
            self.tb_writer.add_figure(name, figure, step)
        
        plt.close(figure)
        logger.info(f"Logged figure: {name}")
    
    def log_table(
        self,
        name: str,
        data: List[Dict],
        columns: Optional[List[str]] = None,
    ):
        """Log tabular data."""
        table_dir = self.exp_dir / "tables"
        table_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        table_path = table_dir / f"{name}.json"
        with open(table_path, "w") as f:
            json.dump(data, f, indent=2)
        
        # Log to W&B
        if self.config.use_wandb:
            columns = columns or list(data[0].keys())
            table = wandb.Table(columns=columns, data=[
                [row.get(c) for c in columns] for row in data
            ])
            wandb.log({name: table})
        
        logger.info(f"Logged table: {name}")
    
    # ========================================================================
    # Experiment Management
    # ========================================================================
    
    def save_summary(self):
        """Save experiment summary."""
        summary = {
            "name": self.config.experiment_name,
            "project": self.config.project,
            "description": self.config.description,
            "tags": self.config.tags,
            "params": self.params,
            "final_metrics": {
                name: history[-1]["value"] if history else None
                for name, history in self.metrics_history.items()
            },
            "artifacts": self.artifacts,
            "started_at": self.metrics_history.get("_start_time", datetime.now().isoformat()),
            "finished_at": datetime.now().isoformat(),
        }
        
        summary_file = self.exp_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def finish(self, status: str = "completed"):
        """Finish experiment tracking."""
        # Save final summary
        summary = self.save_summary()
        
        # Save metrics history
        history_file = self.exp_dir / "metrics_history.json"
        with open(history_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Close backends
        if self.config.use_wandb:
            wandb.finish()
        
        if self.config.use_mlflow:
            mlflow.end_run(status=status)
        
        if self.tb_writer:
            self.tb_writer.close()
        
        logger.info(f"Experiment finished: {self.config.experiment_name}")
        return summary
    
    def get_best_metric(self, name: str, mode: str = "min") -> Dict:
        """Get best value of a metric."""
        if name not in self.metrics_history:
            return {}
        
        history = self.metrics_history[name]
        if not history:
            return {}
        
        if mode == "min":
            best = min(history, key=lambda x: x["value"])
        else:
            best = max(history, key=lambda x: x["value"])
        
        return best


# ============================================================================
# Experiment Registry
# ============================================================================

class ExperimentRegistry:
    """
    Registry for managing multiple experiments.
    
    Features:
    - List experiments
    - Compare experiments
    - Load experiment results
    """
    
    def __init__(self, base_dir: str = "./experiments"):
        self.base_dir = Path(base_dir)
    
    def list_experiments(
        self,
        project: Optional[str] = None,
    ) -> List[Dict]:
        """List all experiments."""
        experiments = []
        
        search_dir = self.base_dir / project if project else self.base_dir
        
        for exp_dir in search_dir.rglob("summary.json"):
            with open(exp_dir) as f:
                summary = json.load(f)
                summary["path"] = str(exp_dir.parent)
                experiments.append(summary)
        
        return sorted(experiments, key=lambda x: x.get("started_at", ""), reverse=True)
    
    def load_experiment(self, path: str) -> Dict:
        """Load experiment results."""
        exp_dir = Path(path)
        
        result = {}
        
        # Load summary
        summary_file = exp_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                result["summary"] = json.load(f)
        
        # Load params
        params_file = exp_dir / "params.json"
        if params_file.exists():
            with open(params_file) as f:
                result["params"] = json.load(f)
        
        # Load metrics history
        history_file = exp_dir / "metrics_history.json"
        if history_file.exists():
            with open(history_file) as f:
                result["metrics_history"] = json.load(f)
        
        return result
    
    def compare_experiments(
        self,
        experiment_paths: List[str],
        metrics: List[str],
    ) -> Dict:
        """Compare multiple experiments."""
        comparison = {
            "experiments": [],
            "metrics": {m: [] for m in metrics},
        }
        
        for path in experiment_paths:
            exp = self.load_experiment(path)
            comparison["experiments"].append({
                "path": path,
                "name": exp.get("summary", {}).get("name", "unknown"),
                "params": exp.get("params", {}),
            })
            
            # Get final metric values
            history = exp.get("metrics_history", {})
            for metric in metrics:
                if metric in history and history[metric]:
                    value = history[metric][-1]["value"]
                else:
                    value = None
                comparison["metrics"][metric].append(value)
        
        return comparison


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate experiment tracking."""
    import numpy as np
    
    print("=" * 60)
    print("EXPERIMENT TRACKING DEMO")
    print("=" * 60)
    
    # Create tracker
    tracker = ExperimentTracker(
        project="demo_project",
        experiment_name=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Demo experiment for testing tracking",
        tags=["demo", "test"],
        use_tensorboard=HAS_TENSORBOARD,
    )
    
    # Log hyperparameters
    tracker.log_params({
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 10,
        "model_type": "transformer",
        "hidden_dim": 512,
    })
    
    # Simulate training
    print("\nSimulating training...")
    for epoch in range(5):
        # Fake metrics
        loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
        accuracy = 0.5 + epoch * 0.1 + np.random.random() * 0.05
        
        tracker.log_metrics({
            "train/loss": loss,
            "train/accuracy": accuracy,
            "epoch": epoch,
        }, step=epoch)
        
        print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
    
    # Log some artifacts
    print("\nLogging artifacts...")
    
    # Create a dummy model
    import torch
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 5),
    )
    
    tracker.log_model(model, "final_model")
    
    # Finish experiment
    summary = tracker.finish()
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Name: {summary['name']}")
    print(f"Project: {summary['project']}")
    print(f"Params: {summary['params']}")
    print(f"Final metrics: {summary['final_metrics']}")
    print(f"Directory: {tracker.exp_dir}")
    
    # List experiments
    print("\n" + "=" * 60)
    print("EXPERIMENT REGISTRY")
    print("=" * 60)
    
    registry = ExperimentRegistry()
    experiments = registry.list_experiments("demo_project")
    
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments[:5]:
        print(f"  - {exp['name']}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
