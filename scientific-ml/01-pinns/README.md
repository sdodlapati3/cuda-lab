# Physics-Informed Neural Networks (PINNs)

## What are PINNs?

PINNs are neural networks that:
1. Approximate solutions to PDEs
2. Encode physics in the loss function
3. Can solve forward AND inverse problems

```
Traditional NN:   Loss = ||u_pred - u_data||²

PINN:            Loss = ||u_pred - u_data||² + λ||PDE_residual||²
                         ↑ Data loss           ↑ Physics loss
```

## Why PINNs Matter

| Approach | Pros | Cons |
|----------|------|------|
| Numerical (FEM, FDM) | Accurate, well-understood | Slow, mesh-dependent |
| Pure ML | Fast | Needs lots of data, no physics |
| **PINNs** | Fast, physics-constrained | Training can be tricky |

## Mathematical Formulation

Consider a PDE of the form:
```
∂u/∂t + N[u] = 0,    x ∈ Ω, t ∈ [0, T]
```

Where N is a nonlinear differential operator.

**PINN Loss Function:**
```
L = L_data + λ₁L_pde + λ₂L_bc + λ₃L_ic
```

- `L_data`: Fit to observed data
- `L_pde`: PDE residual (physics constraint)
- `L_bc`: Boundary condition residual
- `L_ic`: Initial condition residual

## Examples in This Module

1. **[1D Heat Equation](./examples/1d-heat-equation.ipynb)** - Beginner
   - `∂u/∂t = α ∂²u/∂x²`
   - Dirichlet boundary conditions

2. **[Burgers Equation](./examples/burgers-equation.ipynb)** - Intermediate
   - `∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²`
   - Shock wave formation

3. **[Navier-Stokes 2D](./examples/navier-stokes-2d.ipynb)** - Advanced
   - Incompressible flow
   - Lid-driven cavity

## Key Implementation Details

### Automatic Differentiation

```python
def compute_pde_residual(model, x, t):
    """Compute PDE residual using autograd."""
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    
    u = model(x, t)
    
    # First derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), 
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), 
                               create_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), 
                                create_graph=True)[0]
    
    # Heat equation: u_t - α*u_xx = 0
    alpha = 0.1
    residual = u_t - alpha * u_xx
    
    return residual
```

### Loss Balancing

```python
class AdaptiveLossBalancing:
    """Dynamically balance loss components."""
    
    def __init__(self, n_losses):
        self.weights = torch.ones(n_losses)
    
    def update(self, losses):
        # Normalize by gradient magnitudes
        grads = [torch.autograd.grad(l, params)[0].norm() for l in losses]
        max_grad = max(grads)
        self.weights = torch.tensor([max_grad / g for g in grads])
```

### Sampling Strategies

```python
# Uniform sampling in domain
x_pde = torch.rand(n_points) * (x_max - x_min) + x_min
t_pde = torch.rand(n_points) * t_max

# Latin Hypercube Sampling (better coverage)
from scipy.stats import qmc
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=n_points)
```

## Common Challenges

1. **Training instability**: Use learning rate scheduling
2. **Loss imbalance**: Adaptive weighting or gradient balancing
3. **Sharp gradients**: Increase resolution near discontinuities
4. **Slow convergence**: Use better architectures (ResNet, attention)

## Architecture Recommendations

| PDE Type | Recommended Architecture |
|----------|-------------------------|
| Smooth solutions | MLP with Tanh/Swish |
| Sharp gradients | Fourier features + MLP |
| Multi-scale | Modified MLP with skip connections |
| High-dimensional | DeepONet or FNO |

## Exercises

1. **[ex01-custom-pinn](./exercises/ex01-custom-pinn/)** - Implement PINN for your own PDE
   - Choose a PDE from physics/engineering
   - Implement forward problem
   - Validate against analytical solution

## Resources

- [Original PINN Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125) (Raissi et al., 2019)
- [DeepXDE Documentation](https://deepxde.readthedocs.io/)
- [NVIDIA Modulus](https://developer.nvidia.com/modulus)
