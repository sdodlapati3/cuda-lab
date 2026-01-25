# Production ML Infrastructure

This module provides production-ready patterns and tools for deploying
distributed machine learning systems at scale.

## Overview

Moving from research to production requires:

- **Monitoring**: Track training metrics, system health, and costs
- **Fault Tolerance**: Handle failures gracefully with checkpointing
- **MLOps**: Experiment tracking, model versioning, CI/CD
- **Serving**: Deploy models for low-latency inference

## Module Structure

```
production-ml/
├── monitoring/              # Metrics and observability
│   ├── training_monitor.py  # Real-time training metrics
│   └── gpu_monitor.py       # GPU utilization tracking
├── fault-tolerance/         # Resilience patterns
│   ├── elastic_training.py  # Handle node failures
│   └── checkpoint_manager.py # Robust checkpointing
└── mlops/                   # ML operations
    └── experiment_tracking.py # Track experiments
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ML PLATFORM                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    ORCHESTRATION LAYER                           │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │    │
│  │  │    SLURM     │ │  Kubernetes  │ │    Airflow   │             │    │
│  │  │   (HPC)      │ │  (Cloud)     │ │  (Pipelines) │             │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    TRAINING LAYER                                │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │    │
│  │  │     DDP      │ │    FSDP      │ │  DeepSpeed   │             │    │
│  │  │  Multi-GPU   │ │  Sharded     │ │   ZeRO       │             │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    MONITORING LAYER                              │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │    │
│  │  │  Prometheus  │ │   Grafana    │ │    Weights   │             │    │
│  │  │   Metrics    │ │  Dashboard   │ │   & Biases   │             │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    STORAGE LAYER                                 │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │    │
│  │  │  Checkpoint  │ │   Dataset    │ │    Model     │             │    │
│  │  │   Storage    │ │   Storage    │ │   Registry   │             │    │
│  │  └──────────────┘ └──────────────┘ └──────────────┘             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Production Considerations

### 1. Monitoring

| Metric Category | Examples | Tools |
|----------------|----------|-------|
| Training | Loss, LR, throughput | W&B, TensorBoard |
| System | GPU util, memory, I/O | Prometheus, DCGM |
| Cost | GPU hours, cloud spend | Custom dashboards |
| Data | Distribution drift, quality | Great Expectations |

### 2. Fault Tolerance

```
┌────────────────────────────────────────────────────────────────────┐
│                    FAULT TOLERANCE PATTERNS                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Checkpointing Strategy:                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Every N steps:  Save sharded checkpoint                    │   │
│  │  Every epoch:    Save full checkpoint                       │   │
│  │  Best model:     Save when validation improves              │   │
│  │  Keep last K:    Delete old checkpoints                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Failure Recovery:                                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. Node failure → Elastic training resumes                 │   │
│  │  2. OOM → Reduce batch size, retry                         │   │
│  │  3. NaN loss → Rollback to last good checkpoint            │   │
│  │  4. Timeout → Increase timeout, check network              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 3. MLOps Pipeline

```
Code Commit → CI/CD → Training Job → Evaluation → Model Registry → Deployment
     │           │          │            │              │              │
     │           │          │            │              │              ▼
     │           │          │            │              │        A/B Testing
     │           │          │            │              ▼
     │           │          │            │        Versioning
     │           │          │            ▼
     │           │          │        Metrics Logging
     │           │          ▼
     │           │     Checkpoints, Artifacts
     │           ▼
     │     Tests, Linting
     ▼
Git History
```

## Quick Start

### Set Up Monitoring

```python
from production_ml.monitoring import TrainingMonitor

monitor = TrainingMonitor(
    project="my-project",
    experiment="exp-001",
    metrics_port=8000,
)

# In training loop
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    monitor.log({
        "train/loss": loss,
        "train/step": step,
        "system/gpu_util": get_gpu_util(),
    })
```

### Set Up Checkpointing

```python
from production_ml.fault_tolerance import CheckpointManager

ckpt_manager = CheckpointManager(
    checkpoint_dir="/checkpoints",
    keep_last_n=3,
    save_best=True,
)

# Save checkpoint
ckpt_manager.save(
    model=model,
    optimizer=optimizer,
    step=step,
    metrics={"loss": loss, "accuracy": acc},
)

# Resume training
state = ckpt_manager.load_latest(model, optimizer)
start_step = state["step"]
```

### Enable Elastic Training

```bash
# Launch with torchrun elastic mode
torchrun \
    --nnodes=1:4 \
    --nproc_per_node=4 \
    --max_restarts=3 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=head_node:29400 \
    train.py
```

## Best Practices

### 1. Logging
- Log everything: configs, metrics, system stats
- Use structured logging (JSON)
- Aggregate across ranks to rank 0

### 2. Checkpointing
- Save frequently (every 15-30 min)
- Use async saving to not block training
- Keep checkpoint metadata (step, metrics)

### 3. Error Handling
- Catch and log all exceptions
- Implement retry logic with backoff
- Set up alerts for critical failures

### 4. Cost Optimization
- Use spot/preemptible instances with checkpointing
- Right-size GPU allocation
- Implement early stopping

## Resources

- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [Weights & Biases](https://wandb.ai/)
- [MLflow](https://mlflow.org/)
- [Kubernetes Training Operator](https://github.com/kubeflow/training-operator)
