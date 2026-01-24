# Scientific ML Module

> **NESAP Relevance:** "NESAP projects almost always couple ML to simulations or scientific pipelines."

## Overview

Scientific Machine Learning (SciML) combines:
- **Physics** (governing equations, constraints)
- **Data** (observations, experiments)
- **Neural Networks** (function approximation)

This module covers techniques essential for HPC scientific computing roles.

## Contents

1. **[Physics-Informed Neural Networks (PINNs)](./01-pinns/)**
   - Solve PDEs with neural networks
   - Encode physics in loss functions
   
2. **[Surrogate Models](./02-surrogate-models/)**
   - Replace expensive simulations with ML
   - Neural operators (DeepONet, FNO)
   
3. **[Hybrid Solvers](./03-hybrid-solvers/)**
   - ML-enhanced numerical methods
   - Learned preconditioners
   
4. **[Uncertainty Quantification](./04-uncertainty-quantification/)**
   - Bayesian neural networks
   - Ensemble methods
   - Confidence intervals

5. **[Case Studies](./05-case-studies/)**
   - Climate modeling
   - Materials science
   - Particle physics

## Why Scientific ML for NESAP?

| Application | Traditional | Scientific ML |
|-------------|-------------|---------------|
| Weather prediction | Numerical solvers (slow) | Hybrid models (fast + accurate) |
| Molecular dynamics | Ab initio (expensive) | ML potentials (1000x faster) |
| CFD | Mesh-based (hours) | Neural surrogate (seconds) |

## Prerequisites

- Deep learning fundamentals
- Basic PDEs (heat equation, wave equation)
- PyTorch proficiency

## Learning Path

```
Week 1: PINNs fundamentals
        └── 1D heat equation
        └── Loss function design
        
Week 2: Advanced PINNs
        └── Burgers equation
        └── Inverse problems
        
Week 3: Surrogate models
        └── DeepONet
        └── Fourier Neural Operator
        
Week 4: Hybrid methods + UQ
        └── ML preconditioners
        └── Bayesian approaches
```

## Key Libraries

- **[PyTorch](https://pytorch.org/)** - Base framework
- **[DeepXDE](https://deepxde.readthedocs.io/)** - PINNs framework
- **[Modulus](https://developer.nvidia.com/modulus)** - NVIDIA's physics-ML platform
- **[Neural Operator](https://github.com/neuraloperator/neuraloperator)** - FNO implementations
- **[JAX](https://github.com/google/jax)** - Alternative framework (popular in scientific computing)

## Quick Start

```python
import torch
import torch.nn as nn

class PINN(nn.Module):
    """Physics-Informed Neural Network for 1D heat equation."""
    
    def __init__(self, layers=[2, 64, 64, 1]):
        super().__init__()
        self.net = self._build_net(layers)
    
    def _build_net(self, layers):
        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                net.append(nn.Tanh())
        return nn.Sequential(*net)
    
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))
    
    def physics_loss(self, x, t):
        """PDE residual: u_t = alpha * u_xx"""
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        
        # Compute derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0]
        
        # Heat equation: u_t = alpha * u_xx
        alpha = 0.1
        residual = u_t - alpha * u_xx
        
        return torch.mean(residual**2)
```

## NESAP Interview Preparation

Be ready to discuss:
1. "How would you validate a surrogate model?"
2. "What are the trade-offs of PINNs vs traditional solvers?"
3. "How do you handle uncertainty in ML predictions?"
4. "When should you NOT use ML for scientific computing?"
