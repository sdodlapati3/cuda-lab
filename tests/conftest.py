"""
conftest.py - Shared pytest fixtures and configuration

Provides common fixtures for:
- GPU device handling
- Temporary directories
- Sample data generation
- Model creation utilities
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: Quick smoke tests")
    config.addinivalue_line("markers", "gpu: Tests requiring CUDA GPU")
    config.addinivalue_line("markers", "slow: Long-running tests")
    config.addinivalue_line("markers", "integration: End-to-end integration tests")


def pytest_collection_modifyitems(config, items):
    """Automatically add gpu marker to tests using cuda fixtures."""
    for item in items:
        if 'cuda_device' in item.fixturenames:
            item.add_marker(pytest.mark.gpu)


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def cuda_device(cuda_available):
    """Get CUDA device, skip if not available."""
    if not cuda_available:
        pytest.skip("CUDA not available")
    return torch.device('cuda:0')


@pytest.fixture
def device(cuda_available):
    """Get best available device."""
    if cuda_available:
        return torch.device('cuda:0')
    return torch.device('cpu')


@pytest.fixture
def multi_gpu_available():
    """Check if multiple GPUs are available."""
    return torch.cuda.is_available() and torch.cuda.device_count() > 1


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file path."""
    return temp_dir / "test_output.txt"


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def sample_tensor_2d(device):
    """Create a sample 2D tensor."""
    return torch.randn(64, 768, device=device)


@pytest.fixture
def sample_tensor_3d(device):
    """Create a sample 3D tensor (batch, seq, hidden)."""
    return torch.randn(8, 256, 768, device=device)


@pytest.fixture
def sample_image_batch(device):
    """Create a sample batch of images."""
    return torch.randn(32, 3, 224, 224, device=device)


@pytest.fixture
def sample_attention_inputs(device):
    """Create sample Q, K, V for attention tests."""
    batch_size, n_heads, seq_len, head_dim = 8, 12, 256, 64
    Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    return Q, K, V


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def simple_mlp(device):
    """Create a simple MLP model."""
    model = torch.nn.Sequential(
        torch.nn.Linear(768, 3072),
        torch.nn.GELU(),
        torch.nn.Linear(3072, 768),
    ).to(device)
    return model


@pytest.fixture
def simple_cnn(device):
    """Create a simple CNN model."""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 10),
    ).to(device)
    return model


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def repo_root():
    """Get the repository root directory."""
    # Assuming tests are in tests/ directory
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def benchmarks_dir(repo_root):
    """Get the benchmarks directory."""
    return repo_root / "benchmarks"


@pytest.fixture(scope="session")
def profiling_lab_dir(repo_root):
    """Get the profiling-lab directory."""
    return repo_root / "profiling-lab"


@pytest.fixture(scope="session")
def hpc_lab_dir(repo_root):
    """Get the hpc-lab directory."""
    return repo_root / "hpc-lab"


@pytest.fixture(scope="session")
def scientific_ml_dir(repo_root):
    """Get the scientific-ml directory."""
    return repo_root / "scientific-ml"


# =============================================================================
# Utility Functions
# =============================================================================

def assert_tensors_close(a, b, rtol=1e-5, atol=1e-5):
    """Assert two tensors are close."""
    assert torch.allclose(a, b, rtol=rtol, atol=atol), \
        f"Tensors not close. Max diff: {(a - b).abs().max().item()}"


def assert_valid_probability(tensor, dim=-1):
    """Assert tensor is a valid probability distribution."""
    # Sum to 1
    sums = tensor.sum(dim=dim)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        "Probabilities don't sum to 1"
    
    # Non-negative
    assert (tensor >= 0).all(), "Negative probabilities found"
    
    # At most 1
    assert (tensor <= 1).all(), "Probabilities > 1 found"


# =============================================================================
# Skip Conditions
# =============================================================================

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_multi_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 1),
    reason="Multiple GPUs not available"
)

requires_bf16 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="BF16 not supported"
)

requires_flash_attn = pytest.mark.skipif(
    not hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
    reason="Flash Attention not available"
)
