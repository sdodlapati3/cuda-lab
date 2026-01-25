# MLOps

Machine Learning Operations tools for production ML systems.

## Overview

MLOps bridges the gap between ML development and production deployment:
- **Experiment Tracking** - Track hyperparameters, metrics, and artifacts
- **Reproducibility** - Ensure experiments can be reproduced
- **Model Management** - Version, deploy, and monitor models
- **Pipeline Orchestration** - Automate training workflows

## Components

### 1. Experiment Tracking (`experiment_tracking.py`)

Unified experiment tracking with multiple backend support:

```python
from experiment_tracking import ExperimentTracker

tracker = ExperimentTracker(
    project="my_project",
    experiment_name="baseline_v1",
    use_wandb=True,      # Optional W&B integration
    use_mlflow=True,     # Optional MLflow integration
    use_tensorboard=True # TensorBoard integration
)

# Log hyperparameters
tracker.log_params({
    "learning_rate": 1e-4,
    "batch_size": 32,
    "model": "transformer",
})

# Log metrics during training
for step in range(num_steps):
    loss = train_step()
    tracker.log_metrics({"loss": loss}, step=step)

# Log artifacts
tracker.log_model(model, "final_model")
tracker.log_artifact("config.yaml")

# Finish tracking
tracker.finish()
```

### 2. Experiment Registry

Manage and compare multiple experiments:

```python
from experiment_tracking import ExperimentRegistry

registry = ExperimentRegistry("./experiments")

# List all experiments
experiments = registry.list_experiments("my_project")

# Compare experiments
comparison = registry.compare_experiments(
    experiment_paths=[exp1_path, exp2_path],
    metrics=["loss", "accuracy"]
)
```

## Features

### Reproducibility
- Automatically saves git commit hash
- Records Python environment and packages
- Saves copy of training script

### Artifact Management
- Save model checkpoints
- Store configuration files
- Track figures and visualizations

### Backend Integration
- **TensorBoard** - Built-in support
- **Weights & Biases** - `pip install wandb`
- **MLflow** - `pip install mlflow`

## Best Practices

### 1. Always Log Hyperparameters
```python
# Do this at the start of training
tracker.log_params({
    "model_config": model_config,
    "training_config": training_config,
    "data_config": data_config,
})
```

### 2. Use Meaningful Experiment Names
```python
# Good: Descriptive name with version
experiment_name = "transformer_base_v2_longer_training"

# Bad: Generic name
experiment_name = "experiment_1"
```

### 3. Track Everything
```python
# Metrics
tracker.log_metrics({
    "train/loss": loss,
    "train/accuracy": accuracy,
    "train/learning_rate": scheduler.get_lr()[0],
    "system/gpu_memory": torch.cuda.memory_allocated(),
})

# Artifacts
tracker.log_artifact("config.yaml")
tracker.log_model(model, "checkpoint_epoch_10")
```

### 4. Compare Experiments
```python
# After running multiple experiments
registry = ExperimentRegistry()
comparison = registry.compare_experiments(
    experiment_paths=["exp1", "exp2", "exp3"],
    metrics=["final_loss", "best_accuracy"]
)

# Find best performing experiment
best_exp = min(comparison["experiments"], 
               key=lambda x: comparison["metrics"]["final_loss"][
                   comparison["experiments"].index(x)
               ])
```

## Directory Structure

```
experiments/
├── project_name/
│   ├── experiment_1/
│   │   ├── summary.json      # Experiment summary
│   │   ├── params.json       # Hyperparameters
│   │   ├── metrics_history.json
│   │   ├── environment.json  # Python env info
│   │   ├── code/             # Code snapshot
│   │   ├── artifacts/        # Logged artifacts
│   │   ├── models/           # Model checkpoints
│   │   ├── figures/          # Logged figures
│   │   └── tensorboard/      # TensorBoard logs
│   └── experiment_2/
│       └── ...
└── another_project/
    └── ...
```

## Integration Example

Complete training loop with experiment tracking:

```python
from experiment_tracking import ExperimentTracker

def train(config):
    # Initialize tracking
    tracker = ExperimentTracker(
        project=config.project,
        experiment_name=config.experiment_name,
        tags=config.tags,
    )
    tracker.log_params(vars(config))
    
    # Setup model, optimizer, etc.
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    
    # Training loop
    best_loss = float("inf")
    for epoch in range(config.epochs):
        for batch_idx, batch in enumerate(train_loader):
            loss = train_step(model, batch, optimizer)
            step = epoch * len(train_loader) + batch_idx
            
            tracker.log_metrics({
                "train/loss": loss,
                "train/epoch": epoch,
            }, step=step)
        
        # Validation
        val_loss = validate(model, val_loader)
        tracker.log_metrics({
            "val/loss": val_loss,
        }, step=step)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            tracker.log_model(model, "best_model")
    
    # Save final model
    tracker.log_model(model, "final_model")
    tracker.finish()
```
