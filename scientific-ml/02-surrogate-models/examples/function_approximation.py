"""
function_approximation.py - Basic Neural Network Surrogate Model

Learn to approximate a complex function from simulation data.
This is the foundation for all surrogate modeling.

Example: Approximate the 2D Rosenbrock function
    f(x, y) = (a - x)^2 + b(y - x^2)^2

Author: CUDA Lab
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
import argparse


class MLPSurrogate(nn.Module):
    """
    Multi-layer perceptron surrogate model.
    
    Architecture choices:
    - Wider networks: Better for smooth functions
    - Deeper networks: Better for complex patterns
    - Residual connections: Helps gradient flow
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [64, 128, 128, 64],
        activation: str = 'gelu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Activation function
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
        }
        self.activation = activations.get(activation, nn.GELU())
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualSurrogate(nn.Module):
    """
    Residual MLP for better gradient flow.
    Good for deeper networks and complex functions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_blocks: int = 4
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        
        return self.output_proj(x)


def rosenbrock(x: torch.Tensor, a: float = 1.0, b: float = 100.0) -> torch.Tensor:
    """
    Rosenbrock function - classic optimization test function.
    Has a narrow curved valley that's hard to approximate.
    """
    return (a - x[:, 0])**2 + b * (x[:, 1] - x[:, 0]**2)**2


def rastrigin(x: torch.Tensor, A: float = 10.0) -> torch.Tensor:
    """
    Rastrigin function - highly multimodal.
    Tests ability to capture oscillatory behavior.
    """
    n = x.shape[1]
    return A * n + (x**2 - A * torch.cos(2 * np.pi * x)).sum(dim=1)


def generate_training_data(
    func: Callable,
    bounds: Tuple[float, float],
    n_samples: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate training data from ground truth function."""
    X = torch.rand(n_samples, 2, device=device) * (bounds[1] - bounds[0]) + bounds[0]
    y = func(X).unsqueeze(1)
    return X, y


def train_surrogate(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 1000,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = True
) -> dict:
    """
    Train surrogate model with early stopping.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        # Shuffle
        perm = torch.randperm(n_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val).item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 100:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    return history


def evaluate_surrogate(
    model: nn.Module,
    func: Callable,
    bounds: Tuple[float, float],
    n_test: int = 10000,
    device: torch.device = None
) -> dict:
    """Evaluate surrogate model accuracy."""
    if device is None:
        device = next(model.parameters()).device
    
    X_test = torch.rand(n_test, 2, device=device) * (bounds[1] - bounds[0]) + bounds[0]
    y_true = func(X_test)
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).squeeze()
    
    # Metrics
    mse = ((y_pred - y_true)**2).mean().item()
    mae = (y_pred - y_true).abs().mean().item()
    rmse = np.sqrt(mse)
    
    # Relative error
    rel_error = ((y_pred - y_true).abs() / (y_true.abs() + 1e-8)).mean().item()
    
    # R² score
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    r2 = 1 - (ss_res / ss_tot).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'relative_error': rel_error,
        'r2': r2,
    }


def plot_comparison(
    model: nn.Module,
    func: Callable,
    bounds: Tuple[float, float],
    device: torch.device,
    title: str = "Surrogate vs Ground Truth"
):
    """Visualize surrogate approximation quality."""
    n_grid = 100
    x = torch.linspace(bounds[0], bounds[1], n_grid, device=device)
    y = torch.linspace(bounds[0], bounds[1], n_grid, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Ground truth
    Z_true = func(points).reshape(n_grid, n_grid)
    
    # Surrogate prediction
    model.eval()
    with torch.no_grad():
        Z_pred = model(points).reshape(n_grid, n_grid)
    
    # Move to CPU for plotting
    X, Y = X.cpu(), Y.cpu()
    Z_true, Z_pred = Z_true.cpu(), Z_pred.cpu()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Ground truth
    im0 = axes[0].contourf(X, Y, Z_true, levels=50, cmap='viridis')
    axes[0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axes[0])
    
    # Surrogate
    im1 = axes[1].contourf(X, Y, Z_pred, levels=50, cmap='viridis')
    axes[1].set_title('Surrogate Prediction')
    plt.colorbar(im1, ax=axes[1])
    
    # Error
    error = (Z_pred - Z_true).abs()
    im2 = axes[2].contourf(X, Y, error, levels=50, cmap='Reds')
    axes[2].set_title('Absolute Error')
    plt.colorbar(im2, ax=axes[2])
    
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('surrogate_comparison.png', dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Function Approximation Surrogate')
    parser.add_argument('--function', type=str, default='rosenbrock',
                       choices=['rosenbrock', 'rastrigin'])
    parser.add_argument('--n-train', type=int, default=5000)
    parser.add_argument('--n-val', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'residual'])
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Select function
    if args.function == 'rosenbrock':
        func = rosenbrock
        bounds = (-2.0, 2.0)
    else:
        func = rastrigin
        bounds = (-5.12, 5.12)
    
    # Generate data
    print(f"\nGenerating {args.n_train} training samples...")
    X_train, y_train = generate_training_data(func, bounds, args.n_train, device)
    X_val, y_val = generate_training_data(func, bounds, args.n_val, device)
    
    # Normalize outputs for stable training
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    
    # Create model
    if args.model == 'mlp':
        model = MLPSurrogate(input_dim=2, output_dim=1).to(device)
    else:
        model = ResidualSurrogate(input_dim=2, output_dim=1).to(device)
    
    print(f"\nModel: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining...")
    history = train_surrogate(
        model, X_train, y_train_norm, X_val, y_val_norm,
        epochs=args.epochs
    )
    
    # Evaluate (need to unnormalize)
    class UnnormalizedModel(nn.Module):
        def __init__(self, model, mean, std):
            super().__init__()
            self.model = model
            self.mean = mean
            self.std = std
        
        def forward(self, x):
            return self.model(x) * self.std + self.mean
    
    model_eval = UnnormalizedModel(model, y_mean, y_std)
    metrics = evaluate_surrogate(model_eval, func, bounds, device=device)
    
    print("\n" + "="*50)
    print("Evaluation Metrics:")
    print("="*50)
    for name, value in metrics.items():
        print(f"  {name:15s}: {value:.6f}")
    
    # Visualize
    if not args.no_plot:
        plot_comparison(model_eval, func, bounds, device, 
                       f"{args.function.title()} - {args.model.upper()} Surrogate")
    
    # Inference speed comparison
    print("\n" + "="*50)
    print("Inference Speed:")
    print("="*50)
    
    n_inference = 100000
    X_bench = torch.rand(n_inference, 2, device=device) * (bounds[1] - bounds[0]) + bounds[0]
    
    # Ground truth timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    import time
    start = time.perf_counter()
    _ = func(X_bench)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    gt_time = time.perf_counter() - start
    
    # Surrogate timing
    model.eval()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(X_bench)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    surrogate_time = time.perf_counter() - start
    
    print(f"  Ground truth:    {gt_time*1000:.2f} ms for {n_inference:,} samples")
    print(f"  Surrogate:       {surrogate_time*1000:.2f} ms for {n_inference:,} samples")
    print(f"  Speedup:         {gt_time/surrogate_time:.1f}x")


if __name__ == "__main__":
    main()
