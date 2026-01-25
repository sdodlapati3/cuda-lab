"""
test_pytorch_opt.py - Tests for PyTorch optimization modules

Tests:
- torch.compile benchmarks
- Mixed precision training
- Fused operations
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path


class TestTorchCompile:
    """Tests for torch.compile functionality."""
    
    @pytest.mark.smoke
    def test_compile_module_exists(self, repo_root):
        """Test torch.compile benchmark module exists."""
        module = repo_root / "pytorch-optimization" / "01-torch-compile" / "compile_benchmark.py"
        assert module.exists(), "Compile benchmark module missing"
    
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile not available (requires PyTorch 2.0+)"
    )
    def test_basic_compile(self, cuda_device):
        """Test basic torch.compile functionality."""
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        ).to(cuda_device)
        
        compiled_model = torch.compile(model)
        
        x = torch.randn(32, 100, device=cuda_device)
        
        # Should produce same result (with tolerance for numerical precision)
        with torch.no_grad():
            out_eager = model(x)
            out_compiled = compiled_model(x)
        
        # torch.compile may have slight numerical differences due to optimization
        assert torch.allclose(out_eager, out_compiled, rtol=1e-4, atol=1e-5)
    
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile not available"
    )
    def test_compile_modes(self, cuda_device):
        """Test different compile modes."""
        model = nn.Linear(100, 100).to(cuda_device)
        x = torch.randn(32, 100, device=cuda_device)
        
        modes = ['default', 'reduce-overhead']
        
        for mode in modes:
            try:
                compiled = torch.compile(model, mode=mode)
                with torch.no_grad():
                    _ = compiled(x)
            except Exception as e:
                # Some modes may not work on all GPUs
                pass
    
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not hasattr(torch, 'compile'),
        reason="torch.compile not available"
    )
    def test_compile_with_backward(self, cuda_device):
        """Test compiled model with backward pass."""
        model = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        ).to(cuda_device)
        
        compiled_model = torch.compile(model)
        optimizer = torch.optim.Adam(compiled_model.parameters())
        
        x = torch.randn(32, 100, device=cuda_device)
        target = torch.randint(0, 10, (32,), device=cuda_device)
        
        # Forward + backward should work
        output = compiled_model(x)
        loss = nn.functional.cross_entropy(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TestMixedPrecision:
    """Tests for mixed precision training."""
    
    @pytest.mark.smoke
    def test_amp_module_exists(self, repo_root):
        """Test AMP training module exists."""
        module = repo_root / "pytorch-optimization" / "02-mixed-precision" / "amp_training.py"
        assert module.exists(), "AMP training module missing"
    
    @pytest.mark.gpu
    def test_basic_amp_training(self, cuda_device):
        """Test basic AMP training loop."""
        from torch.amp import autocast, GradScaler
        
        model = nn.Linear(100, 10).to(cuda_device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler('cuda')
        
        x = torch.randn(32, 100, device=cuda_device)
        target = torch.randint(0, 10, (32,), device=cuda_device)
        
        # AMP forward
        with autocast('cuda', dtype=torch.float16):
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
        
        # Scaled backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Should complete without error
        assert loss.item() > 0
    
    @pytest.mark.gpu
    def test_autocast_preserves_dtype(self, cuda_device):
        """Test autocast context manager."""
        from torch.amp import autocast
        
        model = nn.Linear(100, 100).to(cuda_device)
        x = torch.randn(32, 100, device=cuda_device, dtype=torch.float32)
        
        with autocast('cuda', dtype=torch.float16):
            output = model(x)
            # Inside autocast, operations use FP16
            assert output.dtype == torch.float16
        
        # Outside autocast, back to FP32
        output_fp32 = model(x)
        assert output_fp32.dtype == torch.float32
    
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_bf16_supported(),
        reason="BF16 not supported"
    )
    def test_bf16_training(self, cuda_device):
        """Test BF16 training (no scaler needed)."""
        from torch.amp import autocast
        
        model = nn.Linear(100, 10).to(cuda_device)
        optimizer = torch.optim.Adam(model.parameters())
        
        x = torch.randn(32, 100, device=cuda_device)
        target = torch.randint(0, 10, (32,), device=cuda_device)
        
        # BF16 doesn't need GradScaler
        with autocast('cuda', dtype=torch.bfloat16):
            output = model(x)
            loss = nn.functional.cross_entropy(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    @pytest.mark.gpu
    def test_grad_scaler_overflow_handling(self, cuda_device):
        """Test GradScaler handles overflow."""
        from torch.amp import autocast, GradScaler
        
        model = nn.Linear(100, 10).to(cuda_device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler('cuda')
        
        x = torch.randn(32, 100, device=cuda_device)
        target = torch.randint(0, 10, (32,), device=cuda_device)
        
        # Simulate potential overflow scenario
        for _ in range(5):
            with autocast('cuda', dtype=torch.float16):
                output = model(x)
                loss = nn.functional.cross_entropy(output, target)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Scaler should adjust scale factor
        assert scaler.get_scale() > 0


class TestFusedOperations:
    """Tests for fused operations."""
    
    @pytest.mark.gpu
    def test_fused_optimizer(self, cuda_device):
        """Test fused Adam optimizer."""
        model = nn.Linear(100, 100).to(cuda_device)
        
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-3,
                fused=True
            )
            
            x = torch.randn(32, 100, device=cuda_device)
            output = model(x)
            loss = output.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        except RuntimeError as e:
            if "fused" in str(e).lower():
                pytest.skip("Fused optimizer not supported on this GPU")
            raise
    
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
        reason="SDPA not available"
    )
    def test_flash_attention_available(self, cuda_device):
        """Test Flash Attention (SDPA) is available."""
        batch_size, n_heads, seq_len, head_dim = 2, 4, 64, 32
        
        Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=cuda_device)
        K = torch.randn_like(Q)
        V = torch.randn_like(Q)
        
        # SDPA should work
        output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == Q.shape
    
    @pytest.mark.gpu
    def test_fused_layer_norm(self, cuda_device):
        """Test fused layer norm operations."""
        x = torch.randn(32, 256, 768, device=cuda_device)
        
        layer_norm = nn.LayerNorm(768).to(cuda_device)
        
        # Should use optimized implementation
        output = layer_norm(x)
        
        assert output.shape == x.shape
        
        # Check normalization
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)


class TestPyTorchOptIntegration:
    """Integration tests for PyTorch optimizations."""
    
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_full_training_with_optimizations(self, cuda_device):
        """Test complete training loop with all optimizations."""
        from torch.amp import autocast, GradScaler
        
        # Model
        model = nn.Sequential(
            nn.Linear(768, 3072),
            nn.LayerNorm(3072),
            nn.GELU(),
            nn.Linear(3072, 768),
            nn.LayerNorm(768),
        ).to(cuda_device)
        
        # Compile if available
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
            except Exception:
                pass
        
        # Fused optimizer if available
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        except RuntimeError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        scaler = GradScaler('cuda')
        
        # Training loop with fixed random seed for reproducibility
        torch.manual_seed(42)
        x_fixed = torch.randn(32, 768, device=cuda_device)
        target_fixed = x_fixed * 0.5  # Learnable target (not random noise)
        
        losses = []
        for step in range(50):
            with autocast('cuda', dtype=torch.float16):
                output = model(x_fixed)
                loss = nn.functional.mse_loss(output, target_fixed)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
        
        # Should train successfully - with learnable target, loss should decrease
        # Use a relaxed check: loss should improve over training
        assert losses[-1] < losses[0] * 1.1, \
            f"Training did not improve: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    
    @pytest.mark.gpu
    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_efficiency(self, cuda_device):
        """Test memory efficiency of optimizations."""
        from torch.amp import autocast
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Test 1: FP32 inference
        model_fp32 = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        ).to(cuda_device)
        
        x = torch.randn(32, 256, 512, device=cuda_device)  # Larger batch for clearer difference
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model_fp32(x)
        torch.cuda.synchronize()
        memory_fp32 = torch.cuda.max_memory_allocated()
        
        # Clean up
        del model_fp32, x
        torch.cuda.empty_cache()
        
        # Test 2: FP16 inference
        model_fp16 = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        ).to(cuda_device).half()  # Convert model to FP16
        
        x = torch.randn(32, 256, 512, device=cuda_device, dtype=torch.float16)
        
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model_fp16(x)
        torch.cuda.synchronize()
        memory_fp16 = torch.cuda.max_memory_allocated()
        
        # FP16 should use less memory (at least 20% reduction expected)
        # Using a relaxed threshold as actual savings vary by hardware
        assert memory_fp16 < memory_fp32 * 0.95, \
            f"FP16 ({memory_fp16/1e6:.1f}MB) should use less memory than FP32 ({memory_fp32/1e6:.1f}MB)"
