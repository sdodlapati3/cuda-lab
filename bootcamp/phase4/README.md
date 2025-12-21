# Phase 4: Domain Applications (Weeks 17-20)

## Overview

Phase 4 applies your CUDA skills to real-world domains: image processing, AI inference, physics simulation, and integration.

## Prerequisites

Before starting Phase 4, you should be able to:
- Use warp-level primitives effectively
- Integrate cuBLAS, CUB, and Thrust
- Fuse kernels to reduce memory traffic
- Manage memory efficiently with pools and pinned memory

## Weekly Themes

### Week 17: Image Processing
Build GPU-accelerated image processing pipelines.

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Convolution Basics | Naive 2D convolution, border handling |
| 2 | Tiled Convolution | Shared memory for filter coefficients |
| 3 | Separable Filters | Decompose 2D into 1D+1D |
| 4 | Histogram & Equalization | Atomic-based histograms, CDF |
| 5 | Image Resizing | Bilinear, bicubic interpolation |
| 6 | Edge Detection | Sobel, gradient magnitude |

### Week 18: AI Inference
Optimize neural network inference for production.

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Model Loading | Weight management, memory layout |
| 2 | Batched Inference | Dynamic batching, padding |
| 3 | Quantization Basics | FP16, INT8 concepts |
| 4 | Layer Fusion | Fused ops for inference |
| 5 | Memory Planning | Buffer reuse, memory pools |
| 6 | Inference Pipeline | End-to-end optimization |

### Week 19: Physics Simulation
GPU-accelerated physics for games and scientific computing.

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | N-Body Basics | Gravitational simulation, O(N²) baseline |
| 2 | Tiled N-Body | Shared memory for particle tiles |
| 3 | Particle Systems | Position, velocity, forces |
| 4 | Spatial Hashing | Grid-based acceleration |
| 5 | Collision Detection | Broad phase, narrow phase |
| 6 | Fluid Basics | SPH fundamentals |

### Week 20: Integration & Capstone
Bring it all together with Python bindings and a complete project.

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Python Bindings | pybind11, ctypes |
| 2 | Multi-GPU Basics | Data parallelism, peer access |
| 3 | Profiling Workflow | End-to-end optimization |
| 4 | Capstone Planning | Project selection, design |
| 5 | Capstone Implementation | Build your project |
| 6 | Capstone Review | Documentation, benchmarks |

## Mental Models for Phase 4

### Image Processing
```
Image = 2D Grid of Pixels
Convolution = Weighted Sum of Neighborhood
Key Optimization: Reuse neighboring pixels via shared memory
```

### Inference Optimization
```
                    ┌─────────────────┐
                    │ Memory-Bound?   │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼ Yes                         ▼ No
    ┌─────────────────┐           ┌─────────────────┐
    │ Fuse Layers     │           │ Optimize Compute│
    │ Reduce Traffic  │           │ (GEMM tuning)   │
    └─────────────────┘           └─────────────────┘
```

### Physics Simulation
```
Per Frame:
1. Compute forces (most expensive)
2. Integrate positions
3. Handle collisions
4. Update spatial structures

GPU Parallelism: One thread per particle
Optimization: Reduce force calculation with spatial structures
```

## Success Criteria

After Phase 4, you can:
- [ ] Build image processing pipelines with 10x+ CPU speedup
- [ ] Optimize inference with layer fusion and quantization
- [ ] Implement real-time physics simulation
- [ ] Create Python bindings for CUDA code
- [ ] Complete an end-to-end project with benchmarks

## Directory Structure

```
phase4/
├── README.md
├── week17/           # Image Processing
│   ├── README.md
│   ├── day1-convolution-basics/
│   ├── day2-tiled-convolution/
│   ├── day3-separable-filters/
│   ├── day4-histogram/
│   ├── day5-image-resize/
│   └── day6-edge-detection/
├── week18/           # AI Inference
│   ├── README.md
│   ├── day1-model-loading/
│   ├── day2-batched-inference/
│   ├── day3-quantization/
│   ├── day4-layer-fusion/
│   ├── day5-memory-planning/
│   └── day6-inference-pipeline/
├── week19/           # Physics Simulation
│   ├── README.md
│   ├── day1-nbody-basics/
│   ├── day2-tiled-nbody/
│   ├── day3-particle-systems/
│   ├── day4-spatial-hashing/
│   ├── day5-collision-detection/
│   └── day6-fluid-basics/
└── week20/           # Integration & Capstone
    ├── README.md
    ├── day1-python-bindings/
    ├── day2-multi-gpu-basics/
    ├── day3-profiling-workflow/
    ├── day4-capstone-planning/
    ├── day5-capstone-implementation/
    └── day6-capstone-review/
```

## Getting Started

```bash
cd phase4/week17/day1-convolution-basics
./build.sh
./build/convolution_basic
```
