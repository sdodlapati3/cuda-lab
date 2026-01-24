# HPC Containers Module

> **NESAP Relevance:** "Can you create a reproducible environment that runs on NERSC?"

## Overview

HPC systems like NERSC use **Singularity/Apptainer** instead of Docker for:
- Security (no root access required)
- Performance (native GPU access)
- Reproducibility (same container everywhere)

## Why Containers in HPC?

| Challenge | Container Solution |
|-----------|-------------------|
| Software dependencies | Self-contained environment |
| System library versions | Bundled libraries |
| Reproducibility | Same container = same results |
| Portability | Run on laptop → HPC cluster |

## Container Runtimes

| Runtime | HPC Support | Root Required | GPU Support |
|---------|-------------|---------------|-------------|
| Docker | ❌ No | Yes | Native |
| Singularity | ✅ Yes | No | Native |
| Apptainer | ✅ Yes | No | Native |
| Podman | ⚠️ Limited | No | Native |

**Apptainer** is the new name for Singularity (community fork).

## Quick Start

### Build a container:
```bash
# From definition file
singularity build pytorch.sif cuda-pytorch.def

# From Docker Hub
singularity build pytorch.sif docker://pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime
```

### Run with GPU:
```bash
# Interactive shell
singularity shell --nv pytorch.sif

# Run a script
singularity exec --nv pytorch.sif python train.py

# In Slurm job
srun singularity exec --nv pytorch.sif python train.py
```

## Templates

1. [cuda-pytorch.def](./templates/cuda-pytorch.def) - PyTorch with CUDA
2. [cuda-jax.def](./templates/cuda-jax.def) - JAX with CUDA
3. [ml-complete.def](./templates/ml-complete.def) - Full ML stack

## Exercises

1. [ex01-build-container](./exercises/ex01-build-container/) - Build your first container
2. [ex02-gpu-in-container](./exercises/ex02-gpu-in-container/) - Run GPU code
3. [ex03-mpi-container](./exercises/ex03-mpi-container/) - Multi-node with MPI

## Key Concepts

### Definition File Structure
```singularity
Bootstrap: docker
From: nvidia/cuda:12.1.0-devel-ubuntu22.04

%post
    # Install software
    apt-get update && apt-get install -y python3 python3-pip
    pip3 install torch

%environment
    export PATH=/usr/local/bin:$PATH

%runscript
    python3 "$@"
```

### Bind Mounts
```bash
# Mount host directory into container
singularity exec --nv -B /scratch:/data pytorch.sif python train.py --data /data
```

### NERSC-Specific
```bash
# On Perlmutter
shifter --image=pytorch/pytorch:latest python train.py

# With Singularity
singularity exec --nv $SCRATCH/containers/pytorch.sif python train.py
```

## Best Practices

1. **Start from official CUDA base images**
2. **Pin all package versions**
3. **Keep containers small** (remove build deps)
4. **Test locally before HPC**
5. **Store in $SCRATCH** (containers can be large)
