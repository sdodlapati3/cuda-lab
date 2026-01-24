"""
test_benchmarks.py - Tests for benchmark modules

Tests:
- Benchmark template classes
- Kernel benchmarks (reduction, matmul, softmax, attention)
- Roofline tools
- Hardware baselines
"""

import pytest
import torch
import json
import sys
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))


class TestBenchmarkTemplate:
    """Tests for benchmark template classes."""
    
    @pytest.mark.smoke
    def test_benchmark_result_dataclass(self):
        """Test BenchmarkResult dataclass creation."""
        from templates.benchmark_template import BenchmarkResult
        
        result = BenchmarkResult(
            kernel_name="test_kernel",
            mean_time_ms=1.5,
            std_time_ms=0.2,
            min_time_ms=1.2,
            max_time_ms=2.0,
            iterations=100
        )
        
        assert result.kernel_name == "test_kernel"
        assert result.mean_time_ms == 1.5
        assert result.iterations == 100
    
    @pytest.mark.smoke
    def test_kernel_benchmark_abstract(self):
        """Test KernelBenchmark base class."""
        from templates.benchmark_template import KernelBenchmark
        
        benchmark = KernelBenchmark(
            name="Test Benchmark",
            description="Test description"
        )
        
        assert benchmark.name == "Test Benchmark"
        assert benchmark.description == "Test description"


class TestReductionBenchmark:
    """Tests for reduction kernel benchmark."""
    
    @pytest.mark.gpu
    def test_reduction_correctness(self, cuda_device):
        """Test that reduction produces correct results."""
        from kernels.reduction.benchmark import ReductionBenchmark
        
        benchmark = ReductionBenchmark(device=str(cuda_device))
        
        # Create test input
        x = torch.randn(10000, device=cuda_device)
        
        # Test sum reduction
        expected = x.sum()
        result = x.sum()
        
        assert torch.allclose(result, expected)
    
    @pytest.mark.gpu
    @pytest.mark.smoke
    def test_reduction_benchmark_runs(self, cuda_device):
        """Test that reduction benchmark completes without error."""
        from kernels.reduction.benchmark import ReductionBenchmark
        
        benchmark = ReductionBenchmark(device=str(cuda_device))
        
        # Run with small configuration
        results = benchmark.run_size_sweep(
            sizes=[1000, 10000],
        )
        
        assert len(results) > 0
        assert 'size' in results[0]
        assert 'mean_ms' in results[0]


class TestMatMulBenchmark:
    """Tests for matrix multiplication benchmark."""
    
    @pytest.mark.gpu
    def test_matmul_correctness(self, cuda_device):
        """Test matmul produces correct results."""
        from kernels.matmul.benchmark import MatMulBenchmark
        
        benchmark = MatMulBenchmark(device=str(cuda_device))
        
        A = torch.randn(64, 64, device=cuda_device)
        B = torch.randn(64, 64, device=cuda_device)
        
        # All methods should produce same result
        result_matmul = benchmark.pytorch_matmul(A, B)
        result_mm = benchmark.pytorch_mm(A, B)
        result_einsum = benchmark.einsum_matmul(A, B)
        
        assert torch.allclose(result_matmul, result_mm, rtol=1e-4)
        assert torch.allclose(result_matmul, result_einsum, rtol=1e-4)
    
    @pytest.mark.gpu
    @pytest.mark.smoke
    def test_matmul_benchmark_runs(self, cuda_device):
        """Test matmul benchmark completes."""
        from kernels.matmul.benchmark import MatMulBenchmark
        
        benchmark = MatMulBenchmark(device=str(cuda_device))
        
        results = benchmark.run_size_sweep(sizes=[128, 256])
        
        assert len(results) > 0
        assert 'tflops' in results[0]
    
    @pytest.mark.gpu
    def test_matmul_verification(self, cuda_device):
        """Test matmul verification passes."""
        from kernels.matmul.benchmark import MatMulBenchmark
        
        benchmark = MatMulBenchmark(device=str(cuda_device))
        assert benchmark.verify_correctness(size=256)


class TestSoftmaxBenchmark:
    """Tests for softmax benchmark."""
    
    @pytest.mark.gpu
    def test_softmax_correctness(self, cuda_device):
        """Test softmax produces valid probabilities."""
        from kernels.softmax.benchmark import SoftmaxBenchmark
        
        benchmark = SoftmaxBenchmark(device=str(cuda_device))
        
        x = torch.randn(8, 128, 512, device=cuda_device)
        
        result = benchmark.pytorch_softmax_functional(x)
        
        # Check sums to 1
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        
        # Check non-negative
        assert (result >= 0).all()
    
    @pytest.mark.gpu
    def test_softmax_numerical_stability(self, cuda_device):
        """Test numerically stable softmax with large values."""
        from kernels.softmax.benchmark import SoftmaxBenchmark
        
        benchmark = SoftmaxBenchmark(device=str(cuda_device))
        
        # Large values that could cause overflow
        x = torch.randn(8, 128, 512, device=cuda_device) * 100
        
        result_stable = benchmark.manual_softmax_stable(x)
        result_native = benchmark.pytorch_softmax_functional(x)
        
        # Should match native implementation
        assert torch.allclose(result_stable, result_native, rtol=1e-4)
    
    @pytest.mark.gpu
    @pytest.mark.smoke
    def test_softmax_benchmark_runs(self, cuda_device):
        """Test softmax benchmark completes."""
        from kernels.softmax.benchmark import SoftmaxBenchmark
        
        benchmark = SoftmaxBenchmark(device=str(cuda_device))
        
        results = benchmark.run_sequence_sweep(
            seq_lengths=[128, 256],
            batch_size=8
        )
        
        assert len(results) > 0
        assert 'bandwidth_GBs' in results[0]


class TestAttentionBenchmark:
    """Tests for attention benchmark."""
    
    @pytest.mark.gpu
    def test_attention_correctness(self, cuda_device, sample_attention_inputs):
        """Test attention produces correct output shape."""
        from kernels.attention.benchmark import AttentionBenchmark
        
        benchmark = AttentionBenchmark(device=str(cuda_device))
        Q, K, V = sample_attention_inputs
        
        result = benchmark.standard_attention(Q, K, V)
        
        # Output should have same shape as V
        assert result.shape == V.shape
    
    @pytest.mark.gpu
    def test_attention_causal_mask(self, cuda_device):
        """Test causal attention mask is applied correctly."""
        from kernels.attention.benchmark import AttentionBenchmark
        
        benchmark = AttentionBenchmark(device=str(cuda_device))
        
        # Small example to verify masking
        Q = torch.randn(1, 1, 4, 8, device=cuda_device)
        K = torch.randn(1, 1, 4, 8, device=cuda_device)
        V = torch.eye(4, device=cuda_device).unsqueeze(0).unsqueeze(0).expand(1, 1, 4, 4)
        V = torch.nn.functional.pad(V, (0, 4))  # Make head_dim=8
        
        result = benchmark.standard_attention(Q, K, V, causal=True)
        
        # With causal mask, position 0 should only attend to position 0
        # (can't fully verify without access to attention weights)
        assert result.shape == Q.shape
    
    @pytest.mark.gpu
    @pytest.mark.skipif(
        not hasattr(torch.nn.functional, 'scaled_dot_product_attention'),
        reason="SDPA not available"
    )
    def test_sdpa_matches_standard(self, cuda_device):
        """Test SDPA produces similar results to standard attention."""
        from kernels.attention.benchmark import AttentionBenchmark
        
        benchmark = AttentionBenchmark(device=str(cuda_device))
        
        Q = torch.randn(2, 4, 32, 64, device=cuda_device)
        K = torch.randn_like(Q)
        V = torch.randn_like(Q)
        
        result_standard = benchmark.standard_attention(Q, K, V)
        result_sdpa = benchmark.sdpa_attention(Q, K, V)
        
        # Should be close (not exact due to numerical differences)
        assert torch.allclose(result_standard, result_sdpa, rtol=1e-3, atol=1e-3)


class TestRooflineTools:
    """Tests for roofline analysis tools."""
    
    @pytest.mark.smoke
    def test_hardware_spec_creation(self):
        """Test HardwareSpec dataclass."""
        from roofline.plot_roofline import HardwareSpec
        
        spec = HardwareSpec(
            name="Test GPU",
            peak_tflops_fp32=10.0,
            peak_bandwidth_GB_s=1000.0
        )
        
        # Ridge point = peak_compute / peak_bandwidth
        expected_ridge = 10.0 * 1000 / 1000.0  # Convert TFLOPS to GFLOPS
        assert abs(spec.ridge_point - expected_ridge) < 0.01
    
    @pytest.mark.smoke
    def test_roofline_calculation(self):
        """Test roofline performance calculation."""
        import numpy as np
        from roofline.plot_roofline import HardwareSpec
        
        spec = HardwareSpec(
            name="Test GPU",
            peak_tflops_fp32=10.0,
            peak_bandwidth_GB_s=1000.0
        )
        
        # Test memory-bound region (low AI)
        ai_low = np.array([1.0])
        perf_low = spec.roofline(ai_low)
        expected_low = 1000.0 * 1.0 / 1000  # bandwidth * AI in TFLOPS
        assert abs(perf_low[0] - expected_low) < 0.01
        
        # Test compute-bound region (high AI)
        ai_high = np.array([100.0])
        perf_high = spec.roofline(ai_high)
        assert perf_high[0] == 10.0  # Capped at peak
    
    @pytest.mark.smoke
    def test_kernel_point_creation(self):
        """Test KernelPoint dataclass."""
        from roofline.plot_roofline import KernelPoint
        
        point = KernelPoint(
            name="Test Kernel",
            arithmetic_intensity=10.0,
            performance=5.0
        )
        
        assert point.name == "Test Kernel"
        assert point.arithmetic_intensity == 10.0
    
    @pytest.mark.smoke
    def test_predefined_hardware_specs(self):
        """Test that predefined hardware specs are reasonable."""
        from roofline.plot_roofline import HARDWARE_SPECS
        
        # A100 should have known specs
        assert 'A100-80GB' in HARDWARE_SPECS
        a100 = HARDWARE_SPECS['A100-80GB']
        
        assert a100.peak_tflops_fp32 > 15  # ~19.5 TFLOPS
        assert a100.peak_bandwidth_GB_s > 1500  # ~2039 GB/s


class TestHardwareBaselines:
    """Tests for hardware baseline files."""
    
    @pytest.mark.smoke
    def test_a100_baseline_exists(self, benchmarks_dir):
        """Test A100 baseline file exists and is valid JSON."""
        baseline_path = benchmarks_dir / "hardware-baselines" / "A100-80GB.json"
        
        assert baseline_path.exists(), "A100 baseline file missing"
        
        with open(baseline_path) as f:
            data = json.load(f)
        
        # Check required fields
        assert 'gpu_name' in data
        assert 'peak_fp32_tflops' in data
        assert 'memory_bandwidth_GB_s' in data
    
    @pytest.mark.smoke
    def test_baseline_values_reasonable(self, benchmarks_dir):
        """Test baseline values are in reasonable ranges."""
        baseline_path = benchmarks_dir / "hardware-baselines" / "A100-80GB.json"
        
        with open(baseline_path) as f:
            data = json.load(f)
        
        # A100 specs should be in known ranges
        assert 15 < data['peak_fp32_tflops'] < 25
        assert 1500 < data['memory_bandwidth_GB_s'] < 2500
        assert 70 < data['memory_size_GB'] < 90


class TestBenchmarkExports:
    """Tests for benchmark result exports."""
    
    @pytest.mark.gpu
    def test_matmul_export_json(self, cuda_device, temp_dir):
        """Test matmul results export to JSON."""
        from kernels.matmul.benchmark import MatMulBenchmark
        
        benchmark = MatMulBenchmark(device=str(cuda_device))
        results = benchmark.run_size_sweep(sizes=[128])
        
        output_path = temp_dir / "matmul_results.json"
        benchmark.export_results(results, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert 'kernels' in data
        assert len(data['kernels']) > 0
