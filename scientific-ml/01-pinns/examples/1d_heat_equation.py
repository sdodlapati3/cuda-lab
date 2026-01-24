"""
1D Heat Equation Solver using Physics-Informed Neural Network

PDE: ∂u/∂t = α ∂²u/∂x²

Domain: x ∈ [0, 1], t ∈ [0, 1]
Initial condition: u(x, 0) = sin(πx)
Boundary conditions: u(0, t) = u(1, t) = 0

Analytical solution: u(x, t) = exp(-α π² t) sin(πx)

Author: CUDA Lab
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class PINN(nn.Module):
    """Physics-Informed Neural Network for 1D heat equation."""
    
    def __init__(
        self,
        layers: list = [2, 64, 64, 64, 1],
        activation: nn.Module = nn.Tanh()
    ):
        """
        Args:
            layers: List of layer sizes [input, hidden..., output]
            activation: Activation function
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.activation = activation
        
        # Initialize weights (Xavier)
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Spatial coordinates [N, 1]
            t: Time coordinates [N, 1]
        
        Returns:
            u: Solution values [N, 1]
        """
        # Concatenate inputs
        inputs = torch.cat([x, t], dim=1)
        
        # Forward through network
        for i, layer in enumerate(self.layers[:-1]):
            inputs = self.activation(layer(inputs))
        
        # Output layer (no activation)
        return self.layers[-1](inputs)


def compute_pde_residual(
    model: PINN,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float = 0.1
) -> torch.Tensor:
    """
    Compute the PDE residual: u_t - α u_xx = 0
    
    Args:
        model: PINN model
        x, t: Collocation points
        alpha: Thermal diffusivity
    
    Returns:
        residual: PDE residual at collocation points
    """
    # Enable gradients for x and t
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    
    # Forward pass
    u = model(x, t)
    
    # Compute u_t (time derivative)
    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # Compute u_x (spatial derivative)
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    # Compute u_xx (second spatial derivative)
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]
    
    # PDE residual: u_t - α u_xx
    residual = u_t - alpha * u_xx
    
    return residual


def analytical_solution(x: np.ndarray, t: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Analytical solution to 1D heat equation.
    
    u(x, t) = exp(-α π² t) sin(πx)
    """
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)


def generate_training_data(
    n_ic: int = 100,    # Initial condition points
    n_bc: int = 100,    # Boundary condition points
    n_pde: int = 10000  # Collocation points for PDE
) -> Tuple[dict, dict, dict]:
    """Generate training data for PINN."""
    
    # Initial condition: u(x, 0) = sin(πx)
    x_ic = torch.rand(n_ic, 1)  # x ∈ [0, 1]
    t_ic = torch.zeros(n_ic, 1)  # t = 0
    u_ic = torch.sin(np.pi * x_ic)  # Initial temperature
    
    # Boundary conditions: u(0, t) = u(1, t) = 0
    t_bc = torch.rand(n_bc, 1)  # t ∈ [0, 1]
    
    # Left boundary: x = 0
    x_bc_left = torch.zeros(n_bc // 2, 1)
    t_bc_left = t_bc[:n_bc // 2]
    u_bc_left = torch.zeros(n_bc // 2, 1)
    
    # Right boundary: x = 1
    x_bc_right = torch.ones(n_bc // 2, 1)
    t_bc_right = t_bc[n_bc // 2:]
    u_bc_right = torch.zeros(n_bc // 2, 1)
    
    # Collocation points for PDE (interior domain)
    x_pde = torch.rand(n_pde, 1)
    t_pde = torch.rand(n_pde, 1)
    
    # Pack into dictionaries
    ic_data = {'x': x_ic, 't': t_ic, 'u': u_ic}
    bc_data = {
        'x': torch.cat([x_bc_left, x_bc_right]),
        't': torch.cat([t_bc_left, t_bc_right]),
        'u': torch.cat([u_bc_left, u_bc_right])
    }
    pde_data = {'x': x_pde, 't': t_pde}
    
    return ic_data, bc_data, pde_data


def train_pinn(
    model: PINN,
    ic_data: dict,
    bc_data: dict,
    pde_data: dict,
    epochs: int = 5000,
    lr: float = 1e-3,
    alpha: float = 0.1,
    lambda_ic: float = 100.0,
    lambda_bc: float = 100.0,
    lambda_pde: float = 1.0,
    verbose: bool = True
) -> list:
    """
    Train the PINN.
    
    Returns:
        loss_history: Training loss over epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=500, factor=0.5, min_lr=1e-6
    )
    
    # Move data to device
    for data in [ic_data, bc_data, pde_data]:
        for key in data:
            data[key] = data[key].to(device)
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Initial condition loss
        u_ic_pred = model(ic_data['x'], ic_data['t'])
        loss_ic = torch.mean((u_ic_pred - ic_data['u'])**2)
        
        # Boundary condition loss
        u_bc_pred = model(bc_data['x'], bc_data['t'])
        loss_bc = torch.mean((u_bc_pred - bc_data['u'])**2)
        
        # PDE residual loss
        residual = compute_pde_residual(
            model, pde_data['x'], pde_data['t'], alpha
        )
        loss_pde = torch.mean(residual**2)
        
        # Total loss
        loss = lambda_ic * loss_ic + lambda_bc * loss_bc + lambda_pde * loss_pde
        
        # Backprop
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        loss_history.append(loss.item())
        
        if verbose and (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  IC: {loss_ic.item():.6f}, BC: {loss_bc.item():.6f}, PDE: {loss_pde.item():.6f}")
    
    return loss_history


def evaluate_and_plot(model: PINN, alpha: float = 0.1):
    """Evaluate model and compare with analytical solution."""
    model.eval()
    
    # Create evaluation grid
    nx, nt = 100, 100
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    
    # Flatten for model input
    x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device)
    t_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32, device=device)
    
    # Predict
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(nt, nx)
    
    # Analytical solution
    u_exact = analytical_solution(X, T, alpha)
    
    # Error
    error = np.abs(u_pred - u_exact)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # PINN prediction
    im0 = axes[0].contourf(X, T, u_pred, levels=50, cmap='viridis')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title('PINN Prediction')
    plt.colorbar(im0, ax=axes[0])
    
    # Analytical solution
    im1 = axes[1].contourf(X, T, u_exact, levels=50, cmap='viridis')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('Analytical Solution')
    plt.colorbar(im1, ax=axes[1])
    
    # Error
    im2 = axes[2].contourf(X, T, error, levels=50, cmap='hot')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    axes[2].set_title(f'Absolute Error (max: {error.max():.4f})')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('heat_equation_results.png', dpi=150)
    plt.show()
    
    print(f"Maximum error: {error.max():.6f}")
    print(f"Mean error: {error.mean():.6f}")
    
    return u_pred, u_exact, error


if __name__ == "__main__":
    # Hyperparameters
    ALPHA = 0.1  # Thermal diffusivity
    EPOCHS = 5000
    LR = 1e-3
    
    # Create model
    model = PINN(layers=[2, 64, 64, 64, 1]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate training data
    ic_data, bc_data, pde_data = generate_training_data(
        n_ic=100, n_bc=100, n_pde=10000
    )
    
    # Train
    print("\nTraining PINN...")
    loss_history = train_pinn(
        model, ic_data, bc_data, pde_data,
        epochs=EPOCHS, lr=LR, alpha=ALPHA
    )
    
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=150)
    plt.show()
    
    # Evaluate
    print("\nEvaluating model...")
    u_pred, u_exact, error = evaluate_and_plot(model, ALPHA)
