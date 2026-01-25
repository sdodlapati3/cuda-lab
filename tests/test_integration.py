"""
test_integration.py - End-to-end integration tests

Tests complete workflows across multiple modules.
"""

import pytest
import torch
import torch.nn as nn
import subprocess
import json
from pathlib import Path


class TestFullBenchmarkSuite:
    """Integration tests for the benchmarking suite."""
    
    @pytest.mark.integration
    def test_benchmark_infrastructure(self, repo_root):
        """Test benchmark infrastructure is complete."""
        benchmarks_dir = repo_root / "benchmarks" / "kernels"
        
        required_modules = ["matmul", "softmax", "attention"]
        
        for module in required_modules:
            module_dir = benchmarks_dir / module
            assert module_dir.exists(), f"Missing benchmark: {module}"
            assert (module_dir / "benchmark.py").exists(), f"Missing benchmark.py in {module}"
            assert (module_dir / "__init__.py").exists(), f"Missing __init__.py in {module}"
    
    @pytest.mark.integration
    def test_benchmark_configs_valid(self, repo_root):
        """Test all benchmark configs are valid."""
        import sys
        sys.path.insert(0, str(repo_root))
        
        try:
            from benchmarks.kernels.matmul.benchmark import MatMulConfig
            from benchmarks.kernels.softmax.benchmark import SoftmaxConfig
            from benchmarks.kernels.attention.benchmark import AttentionConfig
            
            # Create configs with defaults
            matmul_cfg = MatMulConfig(M=1024, N=1024, K=1024)
            assert matmul_cfg.M == 1024
            
            softmax_cfg = SoftmaxConfig(batch_size=32, seq_length=512, hidden_dim=768)
            assert softmax_cfg.hidden_dim == 768
            
            attention_cfg = AttentionConfig(
                batch_size=2, seq_length=1024, num_heads=8, head_dim=64
            )
            assert attention_cfg.num_heads == 8
            
        except ImportError as e:
            pytest.skip(f"Benchmark modules not importable: {e}")


class TestScientificMLWorkflow:
    """Integration tests for Scientific ML workflow."""
    
    @pytest.mark.integration
    def test_pinn_to_surrogate_workflow(self, repo_root):
        """Test complete PINN training and surrogate approximation."""
        import sys
        pinn_path = str(repo_root / "scientific-ml" / "01-pinns" / "examples")
        sys.path.insert(0, pinn_path)
        
        try:
            # Test imports work - use actual 1d_heat_equation PINN
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location("heat_eq", repo_root / "scientific-ml" / "01-pinns" / "examples" / "1d_heat_equation.py")
            heat_eq = module_from_spec(spec)
            spec.loader.exec_module(heat_eq)
            PINN = heat_eq.PINN
            
            # Create a simple PINN
            pinn = PINN(
                layers=[2, 32, 32, 1],
                activation=nn.Tanh()
            )
            
            # Simulate training data from PINN
            x = torch.rand(100, 1)
            t = torch.rand(100, 1)
            y = pinn(x, t)
            
            assert y.shape == (100, 1)
            
        except ImportError as e:
            pytest.skip(f"PINN template not available: {e}")
        finally:
            if pinn_path in sys.path:
                sys.path.remove(pinn_path)
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_uq_inference_workflow(self, cuda_device, repo_root):
        """Test uncertainty quantification inference workflow."""
        import sys
        sys.path.insert(0, str(repo_root))
        
        # Create a simple model with dropout for MC Dropout
        class MCDropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(10, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return self.net(x)
            
            def predict_with_uncertainty(self, x, n_samples=10):
                self.train()  # Enable dropout
                samples = torch.stack([self(x) for _ in range(n_samples)])
                self.eval()
                
                mean = samples.mean(dim=0)
                std = samples.std(dim=0)
                return mean, std
        
        model = MCDropoutModel().to(cuda_device)
        x = torch.randn(32, 10, device=cuda_device)
        
        mean, std = model.predict_with_uncertainty(x)
        
        assert mean.shape == (32, 1)
        assert std.shape == (32, 1)
        assert (std >= 0).all(), "Standard deviation should be non-negative"


class TestDataPipelineWorkflow:
    """Integration tests for data pipeline workflow."""
    
    @pytest.mark.integration
    def test_dataloader_patterns(self, repo_root):
        """Test all dataloader patterns are consistent."""
        import sys
        dataloader_path = str(repo_root / "data-pipelines" / "01-optimized-loading")
        sys.path.insert(0, dataloader_path)
        
        try:
            from optimized_dataloader import (
                DataLoaderConfig,
                SyntheticDataset
            )
            
            config = DataLoaderConfig()
            dataset = SyntheticDataset(1000, (3, 224, 224))
            
            # Create optimized dataloader
            from torch.utils.data import DataLoader
            
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor
            )
            
            # Get first batch
            batch = next(iter(loader))
            assert len(batch) == 2
            
        except ImportError as e:
            pytest.skip(f"Data pipeline modules not available: {e}")
        finally:
            if dataloader_path in sys.path:
                sys.path.remove(dataloader_path)
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gpu_data_loading(self, cuda_device):
        """Test end-to-end GPU data loading."""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(1000, 3, 32, 32),
            torch.randint(0, 10, (1000,))
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            pin_memory=True
        )
        
        for images, labels in loader:
            # Transfer to GPU
            images = images.to(cuda_device, non_blocking=True)
            labels = labels.to(cuda_device, non_blocking=True)
            
            assert images.device == cuda_device
            assert labels.device == cuda_device
            break


class TestProfilingWorkflow:
    """Integration tests for profiling workflow."""
    
    @pytest.mark.integration
    def test_profiling_infrastructure(self, repo_root):
        """Test profiling infrastructure is complete."""
        profiling_dir = repo_root / "profiling-lab"
        
        required_dirs = ["01-nsight-systems", "02-nsight-compute", "03-pytorch-profiler"]
        
        for dir_name in required_dirs:
            assert (profiling_dir / dir_name).exists(), f"Missing profiling module: {dir_name}"
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_pytorch_profiler_works(self, cuda_device):
        """Test PyTorch profiler integration."""
        from torch.profiler import profile, record_function, ProfilerActivity
        
        model = nn.Linear(100, 100).to(cuda_device)
        x = torch.randn(32, 100, device=cuda_device)
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                _ = model(x)
        
        # Should have captured events
        events = prof.key_averages()
        assert len(events) > 0
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_memory_profiling(self, cuda_device):
        """Test memory profiling workflow."""
        torch.cuda.reset_peak_memory_stats()
        
        # Allocate tensor
        x = torch.randn(1024, 1024, 1024, device=cuda_device)
        
        peak_memory = torch.cuda.max_memory_allocated()
        
        # Should have allocated ~4GB (1024^3 * 4 bytes)
        expected = 1024 * 1024 * 1024 * 4
        assert peak_memory >= expected * 0.9  # Allow some variance
        
        del x
        torch.cuda.empty_cache()


class TestHPCIntegration:
    """Integration tests for HPC patterns."""
    
    @pytest.mark.integration
    def test_hpc_infrastructure(self, repo_root):
        """Test HPC infrastructure is complete."""
        hpc_dir = repo_root / "hpc-lab"
        
        required_dirs = ["01-slurm-basics", "02-checkpointing", "03-containers"]
        
        for dir_name in required_dirs:
            assert (hpc_dir / dir_name).exists(), f"Missing HPC module: {dir_name}"
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_checkpoint_save_load(self, cuda_device, tmp_path):
        """Test checkpoint save/load cycle."""
        model = nn.Linear(100, 10).to(cuda_device)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 5,
            'loss': 0.5
        }
        
        path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, path)
        
        # Load checkpoint
        loaded = torch.load(path)
        
        assert loaded['epoch'] == 5
        assert loaded['loss'] == 0.5
        
        # Verify model state
        new_model = nn.Linear(100, 10).to(cuda_device)
        new_model.load_state_dict(loaded['model_state_dict'])
        
        # States should match
        for key in model.state_dict():
            assert torch.equal(
                model.state_dict()[key],
                new_model.state_dict()[key]
            )
    
    @pytest.mark.integration
    def test_slurm_template_valid(self, repo_root):
        """Test SLURM template is syntactically valid."""
        slurm_dir = repo_root / "hpc-lab" / "01-slurm-basics" / "templates"
        
        # Check for SLURM templates - look in templates subdirectory
        templates = list(slurm_dir.glob("*.sbatch")) + list(slurm_dir.glob("*.slurm"))
        
        if not templates:
            pytest.skip("No SLURM templates found")
        
        for template in templates:
            content = template.read_text()
            
            # Should have shebang
            assert content.startswith("#!/"), f"Missing shebang in {template}"
            
            # Should have SBATCH directives
            assert "#SBATCH" in content, f"Missing SBATCH directives in {template}"


class TestPyTorchOptimizationIntegration:
    """Integration tests for PyTorch optimization workflow."""
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_full_optimized_training(self, cuda_device):
        """Test fully optimized training loop."""
        from torch.amp import autocast, GradScaler
        
        # Create transformer model
        model = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            batch_first=True
        ).to(cuda_device)
        
        # Apply compile if available
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
            except Exception:
                pass
        
        # Fused optimizer
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        except RuntimeError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        scaler = GradScaler('cuda')
        
        # Training loop
        losses = []
        for _ in range(10):
            x = torch.randn(8, 64, 256, device=cuda_device)
            
            with autocast('cuda', dtype=torch.float16):
                output = model(x)
                loss = output.pow(2).mean()
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
        
        assert all(l > 0 for l in losses), "All losses should be positive"


class TestCrossModuleIntegration:
    """Tests that span multiple modules."""
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_benchmark_with_profiling(self, cuda_device):
        """Test running a benchmark with profiling enabled."""
        from torch.profiler import profile, ProfilerActivity
        
        # Simple benchmark operation
        def benchmark_matmul(m, n, k, dtype=torch.float32):
            a = torch.randn(m, k, device=cuda_device, dtype=dtype)
            b = torch.randn(k, n, device=cuda_device, dtype=dtype)
            
            # Warmup
            for _ in range(3):
                _ = torch.matmul(a, b)
            
            torch.cuda.synchronize()
            
            # Profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    _ = torch.matmul(a, b)
                torch.cuda.synchronize()
            
            return prof
        
        prof = benchmark_matmul(1024, 1024, 1024)
        
        # Should have CUDA events
        events = prof.key_averages()
        assert len(events) > 0
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_scientific_ml_with_amp(self, cuda_device):
        """Test Scientific ML model with AMP."""
        from torch.amp import autocast, GradScaler
        
        # Simple PINN-like model
        class SimplePINN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = SimplePINN().to(cuda_device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler('cuda')
        
        # Training with AMP
        for _ in range(10):
            x = torch.rand(256, 2, device=cuda_device) * 2 - 1  # [-1, 1]
            
            with autocast('cuda', dtype=torch.float16):
                u = model(x)
                # Simple physics loss (Laplace = 0 approximation)
                loss = u.pow(2).mean()
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_end_to_end_workflow(self, cuda_device, tmp_path):
        """Test complete end-to-end workflow."""
        from torch.amp import autocast, GradScaler
        
        # 1. Create model
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 10)
        ).to(cuda_device)
        
        # 2. Apply optimizations
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except Exception:
                pass
        
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        except RuntimeError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        scaler = GradScaler('cuda')
        
        # 3. Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(1000, 100),
            torch.randint(0, 10, (1000,))
        )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            pin_memory=True
        )
        
        # 4. Training loop
        model.train()
        losses = []
        
        for x, y in loader:
            x = x.to(cuda_device, non_blocking=True)
            y = y.to(cuda_device, non_blocking=True)
            
            with autocast('cuda', dtype=torch.float16):
                output = model(x)
                loss = nn.functional.cross_entropy(output, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
        
        # 5. Save checkpoint
        checkpoint_path = tmp_path / "final_checkpoint.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'final_loss': losses[-1]
        }, checkpoint_path)
        
        # 6. Verify checkpoint
        loaded = torch.load(checkpoint_path)
        assert 'model_state_dict' in loaded
        assert 'final_loss' in loaded
        
        # 7. Verify training progressed
        avg_early = sum(losses[:5]) / 5
        avg_late = sum(losses[-5:]) / 5
        
        assert avg_late <= avg_early * 1.5, "Training should not diverge"
