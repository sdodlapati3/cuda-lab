# Surrogate Models for Scientific Computing

Neural network surrogate models that replace expensive simulations with fast approximations.

## Overview

Surrogate models are trained to approximate the input-output relationship of complex simulations:
- **10-1000x speedup** over traditional solvers
- **Uncertainty quantification** for reliability
- **Differentiable** - enables gradient-based optimization

## Use Cases in HPC

| Application | Traditional Solver | Surrogate Benefit |
|-------------|-------------------|-------------------|
| CFD | Hours per simulation | Milliseconds inference |
| Molecular dynamics | Days for conformations | Real-time screening |
| Weather | Hours for forecast | Sub-second updates |
| Materials | DFT calculations | Fast property prediction |

## Module Contents

```
02-surrogate-models/
├── README.md
├── examples/
│   ├── function_approximation.py   # Basic function fitting
│   ├── pde_surrogate.py            # PDE solution surrogate  
│   ├── uncertainty_aware.py        # Bayesian/ensemble methods
│   └── autoencoder_dynamics.py     # Latent space dynamics
└── exercises/
    ├── ex01-basic-surrogate/       # Fit 2D function
    ├── ex02-parametric-pde/        # Learn parameter-to-solution map
    └── ex03-uncertainty/           # Quantify prediction confidence
```

## Getting Started

```bash
# Load environment
module load python3

# Install dependencies (if needed)
crun -p ~/envs/cuda-lab pip install torch numpy matplotlib scipy

# Run basic example
crun -p ~/envs/cuda-lab python examples/function_approximation.py

# Run with uncertainty quantification
crun -p ~/envs/cuda-lab python examples/uncertainty_aware.py
```

## Key Concepts

### 1. Function Approximation
Learn mapping f: X → Y from simulation data

### 2. Parametric PDEs
Learn mapping (parameters, x, t) → u(x, t; parameters)

### 3. Uncertainty Quantification
- **Ensemble methods**: Train multiple models, use variance
- **MC Dropout**: Dropout at inference for uncertainty
- **Bayesian NNs**: Full posterior over weights

### 4. Latent Space Methods
- Autoencoder compresses high-dim state
- Learn dynamics in latent space
- Decode for full solution

## References

- [DeepONet: Learning nonlinear operators](https://www.nature.com/articles/s42256-021-00302-5)
- [Fourier Neural Operator](https://arxiv.org/abs/2010.08895)
- [Neural Operator](https://arxiv.org/abs/2108.08481)
