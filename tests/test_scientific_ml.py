"""
test_scientific_ml.py - Tests for Scientific ML modules

Tests:
- PINNs implementation
- Surrogate models
- Uncertainty quantification
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path


class TestPINNs:
    """Tests for Physics-Informed Neural Networks."""
    
    @pytest.mark.smoke
    def test_pinns_directory_exists(self, scientific_ml_dir):
        """Test PINNs directory structure exists."""
        pinns_dir = scientific_ml_dir / "01-pinns"
        
        assert pinns_dir.exists(), "PINNs directory missing"
        assert (pinns_dir / "README.md").exists(), "PINNs README missing"
    
    @pytest.mark.smoke
    def test_heat_equation_example_exists(self, scientific_ml_dir):
        """Test heat equation example exists."""
        example = scientific_ml_dir / "01-pinns" / "examples" / "1d_heat_equation.py"
        assert example.exists(), "Heat equation example missing"
    
    @pytest.mark.gpu
    def test_pinn_forward_pass(self, cuda_device):
        """Test basic PINN forward pass."""
        # Simple PINN architecture
        class SimplePINN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 32),  # x, t inputs
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1)   # u output
                )
            
            def forward(self, x, t):
                inputs = torch.cat([x, t], dim=-1)
                return self.net(inputs)
        
        model = SimplePINN().to(cuda_device)
        
        # Test forward pass
        x = torch.randn(100, 1, device=cuda_device)
        t = torch.randn(100, 1, device=cuda_device)
        
        u = model(x, t)
        
        assert u.shape == (100, 1)
    
    @pytest.mark.gpu
    def test_pinn_gradients(self, cuda_device):
        """Test PINN can compute gradients for physics loss."""
        class PINN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x, t):
                inputs = torch.cat([x, t], dim=-1)
                return self.net(inputs)
        
        model = PINN().to(cuda_device)
        
        # Need gradients w.r.t. inputs
        x = torch.randn(100, 1, device=cuda_device, requires_grad=True)
        t = torch.randn(100, 1, device=cuda_device, requires_grad=True)
        
        u = model(x, t)
        
        # Compute du/dx
        du_dx = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        # Compute du/dt
        du_dt = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        assert du_dx.shape == x.shape
        assert du_dt.shape == t.shape


class TestSurrogateModels:
    """Tests for surrogate model implementations."""
    
    @pytest.mark.smoke
    def test_surrogate_directory_exists(self, scientific_ml_dir):
        """Test surrogate models directory exists."""
        surrogate_dir = scientific_ml_dir / "02-surrogate-models"
        
        assert surrogate_dir.exists(), "Surrogate models directory missing"
        assert (surrogate_dir / "README.md").exists()
    
    @pytest.mark.smoke
    def test_function_approximation_exists(self, scientific_ml_dir):
        """Test function approximation example exists."""
        example = scientific_ml_dir / "02-surrogate-models" / "examples" / "function_approximation.py"
        assert example.exists(), "Function approximation example missing"
    
    @pytest.mark.gpu
    def test_mlp_surrogate(self, cuda_device):
        """Test MLP surrogate model."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scientific-ml" / "02-surrogate-models" / "examples"))
        
        try:
            from function_approximation import MLPSurrogate
            
            model = MLPSurrogate(
                input_dim=2,
                output_dim=1,
                hidden_dims=[32, 32]
            ).to(cuda_device)
            
            x = torch.randn(100, 2, device=cuda_device)
            y = model(x)
            
            assert y.shape == (100, 1)
            
        except ImportError:
            pytest.skip("Could not import MLPSurrogate")
        finally:
            if sys.path[0].endswith("examples"):
                sys.path.pop(0)
    
    @pytest.mark.gpu
    def test_residual_surrogate(self, cuda_device):
        """Test residual surrogate model."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scientific-ml" / "02-surrogate-models" / "examples"))
        
        try:
            from function_approximation import ResidualSurrogate
            
            model = ResidualSurrogate(
                input_dim=2,
                output_dim=1,
                hidden_dim=32,
                num_blocks=2
            ).to(cuda_device)
            
            x = torch.randn(100, 2, device=cuda_device)
            y = model(x)
            
            assert y.shape == (100, 1)
            
        except ImportError:
            pytest.skip("Could not import ResidualSurrogate")
        finally:
            if sys.path[0].endswith("examples"):
                sys.path.pop(0)


class TestUncertaintyQuantification:
    """Tests for uncertainty quantification methods."""
    
    @pytest.mark.smoke
    def test_uq_example_exists(self, scientific_ml_dir):
        """Test UQ example exists."""
        example = scientific_ml_dir / "02-surrogate-models" / "examples" / "uncertainty_aware.py"
        assert example.exists(), "Uncertainty aware example missing"
    
    @pytest.mark.gpu
    def test_mc_dropout_uncertainty(self, cuda_device):
        """Test MC Dropout produces uncertainty estimates."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scientific-ml" / "02-surrogate-models" / "examples"))
        
        try:
            from uncertainty_aware import MCDropoutSurrogate
            
            model = MCDropoutSurrogate(
                input_dim=1,
                output_dim=1,
                hidden_dims=[32, 32],
                dropout_rate=0.1
            ).to(cuda_device)
            
            x = torch.randn(50, 1, device=cuda_device)
            
            mean, std = model.predict_with_uncertainty(x, n_samples=20)
            
            # Should return uncertainty
            assert std.shape == mean.shape
            assert (std >= 0).all()  # Uncertainty is non-negative
            
        except ImportError:
            pytest.skip("Could not import MCDropoutSurrogate")
        finally:
            if sys.path[0].endswith("examples"):
                sys.path.pop(0)
    
    @pytest.mark.gpu
    def test_heteroscedastic_uncertainty(self, cuda_device):
        """Test heteroscedastic model predicts mean and variance."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scientific-ml" / "02-surrogate-models" / "examples"))
        
        try:
            from uncertainty_aware import HeteroscedasticSurrogate
            
            model = HeteroscedasticSurrogate(
                input_dim=1,
                hidden_dims=[32, 32]
            ).to(cuda_device)
            
            x = torch.randn(50, 1, device=cuda_device)
            
            mean, std = model.predict_with_uncertainty(x)
            
            assert mean.shape == (50, 1)
            assert std.shape == (50, 1)
            assert (std > 0).all()  # Variance must be positive
            
        except ImportError:
            pytest.skip("Could not import HeteroscedasticSurrogate")
        finally:
            if sys.path[0].endswith("examples"):
                sys.path.pop(0)
    
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_deep_ensemble_uncertainty(self, cuda_device):
        """Test deep ensemble produces uncertainty."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "scientific-ml" / "02-surrogate-models" / "examples"))
        
        try:
            from uncertainty_aware import DeepEnsemble
            
            ensemble = DeepEnsemble(
                input_dim=1,
                output_dim=1,
                hidden_dims=[32],
                n_members=3,
                device=cuda_device
            )
            
            # Quick training
            X_train = torch.randn(100, 1, device=cuda_device)
            y_train = torch.sin(X_train)
            
            ensemble.train_ensemble(X_train, y_train, epochs=10)
            
            X_test = torch.randn(20, 1, device=cuda_device)
            mean, std = ensemble.predict_with_uncertainty(X_test)
            
            assert mean.shape == (20, 1)
            assert std.shape == (20, 1)
            assert (std >= 0).all()
            
        except ImportError:
            pytest.skip("Could not import DeepEnsemble")
        finally:
            if sys.path[0].endswith("examples"):
                sys.path.pop(0)


class TestScientificMLIntegration:
    """Integration tests for scientific ML workflows."""
    
    @pytest.mark.gpu
    @pytest.mark.integration
    @pytest.mark.slow
    def test_pinn_training_loop(self, cuda_device):
        """Test complete PINN training loop."""
        
        # Simple PINN for 1D problem
        class PINN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.Tanh(),
                    nn.Linear(32, 32),
                    nn.Tanh(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = PINN().to(cuda_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training loop
        losses = []
        for epoch in range(100):
            # Physics points
            x_physics = torch.rand(100, 1, device=cuda_device, requires_grad=True)
            
            u = model(x_physics)
            
            # Simple ODE: du/dx = x
            du_dx = torch.autograd.grad(
                u, x_physics,
                grad_outputs=torch.ones_like(u),
                create_graph=True
            )[0]
            
            physics_loss = ((du_dx - x_physics) ** 2).mean()
            
            # Boundary condition: u(0) = 0
            x_bc = torch.zeros(1, 1, device=cuda_device)
            u_bc = model(x_bc)
            bc_loss = (u_bc ** 2).mean()
            
            loss = physics_loss + bc_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0], "PINN training did not converge"
    
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_surrogate_vs_true_function(self, cuda_device):
        """Test surrogate model approximates true function."""
        
        # True function: f(x) = sin(x)
        def true_function(x):
            return torch.sin(x)
        
        # Create and train surrogate
        model = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(cuda_device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training
        for _ in range(500):
            x = torch.rand(256, 1, device=cuda_device) * 2 * 3.14159 - 3.14159
            y_true = true_function(x)
            
            y_pred = model(x)
            loss = ((y_pred - y_true) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate
        x_test = torch.linspace(-3.14, 3.14, 100, device=cuda_device).unsqueeze(1)
        y_test_true = true_function(x_test)
        
        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
        
        # Should have reasonable approximation
        mse = ((y_test_pred - y_test_true) ** 2).mean().item()
        assert mse < 0.01, f"Surrogate MSE too high: {mse}"
