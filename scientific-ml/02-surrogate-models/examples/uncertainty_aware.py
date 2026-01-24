"""
uncertainty_aware.py - Surrogate Models with Uncertainty Quantification

Knowing when to trust your surrogate is as important as the prediction itself.
This module implements multiple UQ approaches:

1. Deep Ensembles - Train multiple models, use disagreement as uncertainty
2. MC Dropout - Dropout at inference for approximate Bayesian inference
3. Heteroscedastic Networks - Predict mean and variance directly

Author: CUDA Lab
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import argparse


class MCDropoutSurrogate(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Key insight: Dropout at test time ≈ Approximate Bayesian inference
    Multiple forward passes with dropout give uncertainty estimate.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128, 128],
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multiple forward passes with dropout enabled.
        
        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std


class HeteroscedasticSurrogate(nn.Module):
    """
    Network that predicts both mean and variance.
    
    Useful when uncertainty varies across input space
    (heteroscedastic noise).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 128, 128]
    ):
        super().__init__()
        
        # Shared backbone
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Log-variance head (log for numerical stability)
        self.logvar_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        return mean, logvar
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean and std."""
        self.eval()
        with torch.no_grad():
            mean, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        return mean, std


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty quantification.
    
    Train multiple models with different initializations.
    Disagreement between models indicates uncertainty.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        n_members: int = 5,
        device: torch.device = None
    ):
        self.n_members = n_members
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create ensemble members
        self.members = []
        for _ in range(n_members):
            model = self._create_member(input_dim, output_dim, hidden_dims)
            model.to(self.device)
            self.members.append(model)
    
    def _create_member(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int]
    ) -> nn.Module:
        """Create a single ensemble member."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def train_ensemble(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 500,
        lr: float = 1e-3,
        batch_size: int = 256,
        bootstrap: bool = True
    ):
        """
        Train all ensemble members.
        
        Args:
            bootstrap: If True, each member sees different subset of data
        """
        n_samples = X_train.shape[0]
        
        for i, model in enumerate(self.members):
            print(f"Training member {i+1}/{self.n_members}")
            
            # Bootstrap sampling
            if bootstrap:
                indices = torch.randint(0, n_samples, (n_samples,))
                X_member = X_train[indices]
                y_member = y_train[indices]
            else:
                X_member = X_train
                y_member = y_train
            
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(epochs):
                # Shuffle
                perm = torch.randperm(n_samples)
                X_member = X_member[perm]
                y_member = y_member[perm]
                
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    
                    optimizer.zero_grad()
                    pred = model(X_member[start:end])
                    loss = criterion(pred, y_member[start:end])
                    loss.backward()
                    optimizer.step()
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ensemble prediction with uncertainty.
        
        Returns:
            mean: Mean of ensemble predictions
            std: Standard deviation (epistemic uncertainty)
        """
        predictions = []
        
        for model in self.members:
            model.eval()
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std


def gaussian_nll_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Negative log-likelihood for Gaussian distribution.
    
    NLL = 0.5 * (log(var) + (y - mean)^2 / var)
    """
    var = torch.exp(logvar)
    return 0.5 * (logvar + (target - mean)**2 / var).mean()


def train_heteroscedastic(
    model: HeteroscedasticSurrogate,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256
):
    """Train heteroscedastic model with NLL loss."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n_samples = X_train.shape[0]
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        total_loss = 0
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            
            optimizer.zero_grad()
            mean, logvar = model(X_train[start:end])
            loss = gaussian_nll_loss(mean, logvar, y_train[start:end])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {total_loss:.4f}")


def train_mc_dropout(
    model: MCDropoutSurrogate,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 500,
    lr: float = 1e-3,
    batch_size: int = 256
):
    """Train MC Dropout model."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    n_samples = X_train.shape[0]
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            
            optimizer.zero_grad()
            pred = model(X_train[start:end])
            loss = criterion(pred, y_train[start:end])
            loss.backward()
            optimizer.step()


def demo_out_of_distribution(device: torch.device):
    """
    Demonstrate uncertainty awareness on out-of-distribution data.
    
    Train on [-2, 2], test on [-4, 4]
    Uncertainty should increase outside training domain.
    """
    print("\n" + "="*60)
    print("Out-of-Distribution Detection Demo")
    print("="*60)
    
    # Training data: sine wave with noise
    def true_function(x):
        return torch.sin(2 * x) + 0.1 * torch.randn_like(x)
    
    X_train = torch.linspace(-2, 2, 200, device=device).unsqueeze(1)
    y_train = true_function(X_train)
    
    # Test data: extended range
    X_test = torch.linspace(-4, 4, 400, device=device).unsqueeze(1)
    y_test_true = torch.sin(2 * X_test)
    
    # Train each method
    results = {}
    
    # 1. MC Dropout
    print("\n1. Training MC Dropout model...")
    mc_model = MCDropoutSurrogate(1, 1, dropout_rate=0.1).to(device)
    train_mc_dropout(mc_model, X_train, y_train, epochs=500)
    mean_mc, std_mc = mc_model.predict_with_uncertainty(X_test, n_samples=50)
    results['MC Dropout'] = (mean_mc, std_mc)
    
    # 2. Heteroscedastic
    print("\n2. Training Heteroscedastic model...")
    hetero_model = HeteroscedasticSurrogate(1).to(device)
    train_heteroscedastic(hetero_model, X_train, y_train, epochs=500)
    mean_hetero, std_hetero = hetero_model.predict_with_uncertainty(X_test)
    results['Heteroscedastic'] = (mean_hetero, std_hetero)
    
    # 3. Deep Ensemble
    print("\n3. Training Deep Ensemble (5 members)...")
    ensemble = DeepEnsemble(1, 1, n_members=5, device=device)
    ensemble.train_ensemble(X_train, y_train, epochs=300)
    mean_ens, std_ens = ensemble.predict_with_uncertainty(X_test)
    results['Deep Ensemble'] = (mean_ens, std_ens)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    X_train_np = X_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test_true.cpu().numpy()
    
    for ax, (name, (mean, std)) in zip(axes, results.items()):
        mean_np = mean.cpu().numpy().flatten()
        std_np = std.cpu().numpy().flatten()
        
        ax.fill_between(
            X_test_np.flatten(),
            mean_np - 2*std_np,
            mean_np + 2*std_np,
            alpha=0.3, label='±2σ'
        )
        ax.plot(X_test_np, mean_np, 'b-', linewidth=2, label='Prediction')
        ax.plot(X_test_np, y_test_np, 'k--', linewidth=1, label='True')
        ax.scatter(X_train_np, y_train_np, c='red', s=10, alpha=0.5, label='Train')
        
        # Mark OOD region
        ax.axvspan(-4, -2, alpha=0.1, color='gray')
        ax.axvspan(2, 4, alpha=0.1, color='gray')
        ax.text(-3, 1.5, 'OOD', fontsize=12, ha='center')
        ax.text(3, 1.5, 'OOD', fontsize=12, ha='center')
        
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='lower left', fontsize=8)
        ax.set_ylim(-2.5, 2.5)
    
    plt.suptitle('Uncertainty Quantification: Out-of-Distribution Detection')
    plt.tight_layout()
    plt.savefig('uq_ood_comparison.png', dpi=150)
    plt.show()
    
    # Print metrics
    print("\n" + "="*60)
    print("OOD Uncertainty Analysis")
    print("="*60)
    
    in_domain = (X_test.abs() <= 2).squeeze()
    out_domain = (X_test.abs() > 2).squeeze()
    
    for name, (mean, std) in results.items():
        std_in = std[in_domain].mean().item()
        std_out = std[out_domain].mean().item()
        ratio = std_out / std_in
        
        print(f"\n{name}:")
        print(f"  In-domain std:  {std_in:.4f}")
        print(f"  OOD std:        {std_out:.4f}")
        print(f"  Ratio (OOD/ID): {ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Uncertainty-Aware Surrogate Models')
    parser.add_argument('--demo', type=str, default='ood',
                       choices=['ood'],
                       help='Demo to run')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if args.demo == 'ood':
        demo_out_of_distribution(device)


if __name__ == "__main__":
    main()
