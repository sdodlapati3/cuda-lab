# Data Pipelines Module

Optimized data loading and I/O patterns for GPU training at HPC scale.

## Overview

Data loading is often the bottleneck in GPU training. This module covers:
- NVIDIA DALI for GPU-accelerated preprocessing
- Efficient file formats (HDF5, Zarr, WebDataset)
- Multi-node data loading strategies
- Memory-mapped data access

## Module Structure

```
data-pipelines/
├── README.md
├── 01-dali-basics/           # NVIDIA DALI introduction
├── 02-efficient-formats/     # HDF5, Zarr, TFRecord
├── 03-distributed-loading/   # Multi-GPU/Multi-node
└── 04-streaming-data/        # WebDataset, infinite streams
```

## Key Concepts

### The Data Loading Bottleneck

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  Disk   │ --> │  CPU    │ --> │  GPU    │ --> │ Model   │
│  I/O    │     │ Decode  │     │ Augment │     │ Forward │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
   Slow           Medium          Fast           Fast
```

**Goal**: Keep GPU utilization high by overlapping I/O with compute.

### Solutions

| Approach | Best For | Speedup |
|----------|----------|---------|
| DataLoader workers | Small files, CPU decode | 2-4x |
| DALI | Images, GPU decode | 3-10x |
| Memory mapping | Random access | 2-5x |
| WebDataset | Sequential streaming | 2-3x |
| Prefetching | Any pipeline | 1.5-2x |

## Getting Started

```bash
# Load environment
module load python3

# Install DALI (requires CUDA)
crun -p ~/envs/cuda-lab pip install nvidia-dali-cuda120

# Install other dependencies  
crun -p ~/envs/cuda-lab pip install h5py zarr webdataset

# Run DALI example
crun -p ~/envs/cuda-lab python 01-dali-basics/image_pipeline.py
```

## Performance Guidelines

1. **Profile first**: Use PyTorch Profiler to identify bottleneck
2. **Increase workers**: Start with 4-8 workers per GPU
3. **Use prefetch**: `prefetch_factor=2` or higher
4. **Pin memory**: `pin_memory=True` for faster transfers
5. **Consider DALI**: If CPU preprocessing is bottleneck

## References

- [NVIDIA DALI Documentation](https://docs.nvidia.com/deeplearning/dali/)
- [WebDataset for Large-Scale Training](https://github.com/webdataset/webdataset)
- [PyTorch DataLoader Best Practices](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
