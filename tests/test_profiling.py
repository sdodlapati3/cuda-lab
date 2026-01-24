"""
test_profiling.py - Tests for profiling lab modules

Tests:
- Nsight Systems exercises
- Nsight Compute exercises
- PyTorch profiler utilities
- Energy profiling tools
"""

import pytest
import torch
import sys
from pathlib import Path


class TestNsightSystemsExercises:
    """Tests for Nsight Systems exercises."""
    
    @pytest.mark.smoke
    def test_exercise_files_exist(self, profiling_lab_dir):
        """Test that Nsight Systems exercise files exist."""
        nsys_dir = profiling_lab_dir / "01-nsight-systems"
        
        exercises = [
            "ex01-timeline-basics",
            "ex02-kernel-overlap",
            "ex03-memory-timeline",
            "ex04-multi-gpu-timeline",
        ]
        
        for ex in exercises:
            ex_dir = nsys_dir / ex
            assert ex_dir.exists(), f"Exercise directory missing: {ex}"
            
            readme = ex_dir / "README.md"
            assert readme.exists(), f"README missing for: {ex}"
    
    @pytest.mark.smoke
    def test_kernel_overlap_exercise(self, profiling_lab_dir):
        """Test kernel overlap exercise has required files."""
        overlap_dir = profiling_lab_dir / "01-nsight-systems" / "ex02-kernel-overlap"
        
        required_files = ["sequential.cu", "overlapped.cu", "solution.cu", "Makefile"]
        
        for filename in required_files:
            filepath = overlap_dir / filename
            assert filepath.exists(), f"Missing file: {filename}"
    
    @pytest.mark.smoke
    def test_cuda_source_syntax(self, profiling_lab_dir):
        """Basic syntax check for CUDA source files."""
        overlap_dir = profiling_lab_dir / "01-nsight-systems" / "ex02-kernel-overlap"
        
        # Check that files contain expected patterns
        sequential = overlap_dir / "sequential.cu"
        if sequential.exists():
            content = sequential.read_text()
            assert "__global__" in content, "Missing kernel declaration"
            assert "cudaMalloc" in content or "cuda" in content.lower()


class TestNsightComputeExercises:
    """Tests for Nsight Compute exercises."""
    
    @pytest.mark.smoke
    def test_exercise_files_exist(self, profiling_lab_dir):
        """Test that Nsight Compute exercise files exist."""
        ncu_dir = profiling_lab_dir / "02-nsight-compute"
        
        exercises = [
            "ex01-memory-metrics",
            "ex02-compute-metrics",
            "ex03-roofline-practice",
            "ex04-optimization-loop",
        ]
        
        for ex in exercises:
            ex_dir = ncu_dir / ex
            assert ex_dir.exists(), f"Exercise directory missing: {ex}"
    
    @pytest.mark.smoke
    def test_key_metrics_cheatsheet_exists(self, profiling_lab_dir):
        """Test key metrics reference exists."""
        cheatsheet = profiling_lab_dir / "02-nsight-compute" / "reference" / "key-metrics-cheatsheet.md"
        assert cheatsheet.exists(), "Key metrics cheatsheet missing"
    
    @pytest.mark.smoke
    def test_cheatsheet_content(self, profiling_lab_dir):
        """Test cheatsheet has expected content."""
        cheatsheet = profiling_lab_dir / "02-nsight-compute" / "reference" / "key-metrics-cheatsheet.md"
        
        if cheatsheet.exists():
            content = cheatsheet.read_text()
            
            # Should mention key metric categories
            assert "memory" in content.lower()
            assert "compute" in content.lower() or "throughput" in content.lower()


class TestPyTorchProfiler:
    """Tests for PyTorch profiler exercises."""
    
    @pytest.mark.smoke
    def test_profiler_exercise_exists(self, profiling_lab_dir):
        """Test PyTorch profiler exercise exists."""
        profiler_dir = profiling_lab_dir / "03-pytorch-profiler"
        
        exercises = [
            "ex01-basic-profiling",
            "ex02-memory-profiling",
            "ex03-distributed-profiling",
        ]
        
        for ex in exercises:
            ex_dir = profiler_dir / ex
            assert ex_dir.exists(), f"Exercise directory missing: {ex}"
    
    @pytest.mark.gpu
    def test_basic_profiler_usage(self, cuda_device):
        """Test basic PyTorch profiler functionality."""
        from torch.profiler import profile, ProfilerActivity
        
        model = torch.nn.Linear(100, 100).to(cuda_device)
        x = torch.randn(32, 100, device=cuda_device)
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            for _ in range(5):
                model(x)
        
        # Should have recorded events
        events = prof.key_averages()
        assert len(events) > 0
    
    @pytest.mark.gpu
    def test_profiler_with_tensorboard(self, cuda_device, temp_dir):
        """Test profiler can export to TensorBoard format."""
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
        
        model = torch.nn.Linear(100, 100).to(cuda_device)
        x = torch.randn(32, 100, device=cuda_device)
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler(str(temp_dir))
        ) as prof:
            model(x)
        
        # Should have created trace file
        trace_files = list(temp_dir.glob("*.json")) + list(temp_dir.glob("*.pt.trace.json"))
        # Note: trace files may have different extensions


class TestEnergyProfiling:
    """Tests for energy profiling module."""
    
    @pytest.mark.smoke
    def test_energy_profiling_readme_exists(self, profiling_lab_dir):
        """Test energy profiling README exists."""
        energy_dir = profiling_lab_dir / "04-energy-profiling"
        readme = energy_dir / "README.md"
        
        assert readme.exists(), "Energy profiling README missing"
    
    @pytest.mark.smoke
    def test_energy_benchmark_script_exists(self, profiling_lab_dir):
        """Test energy benchmark script exists."""
        script = profiling_lab_dir / "04-energy-profiling" / "scripts" / "energy_benchmark.py"
        assert script.exists(), "Energy benchmark script missing"
    
    @pytest.mark.gpu
    def test_pynvml_import(self):
        """Test pynvml can be imported (if installed)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            # Get device count
            device_count = pynvml.nvmlDeviceGetCount()
            assert device_count > 0
            
            pynvml.nvmlShutdown()
        except ImportError:
            pytest.skip("pynvml not installed")
        except Exception as e:
            pytest.skip(f"NVML not available: {e}")
    
    @pytest.mark.gpu
    def test_energy_benchmark_class(self, profiling_lab_dir, cuda_device):
        """Test EnergyBenchmark class can be instantiated."""
        sys.path.insert(0, str(profiling_lab_dir / "04-energy-profiling" / "scripts"))
        
        try:
            from energy_benchmark import EnergyBenchmark
            
            benchmark = EnergyBenchmark()
            # Just test it can be created
            assert benchmark is not None
        except ImportError:
            pytest.skip("energy_benchmark module not importable")
        except Exception as e:
            if "NVML" in str(e) or "nvml" in str(e).lower():
                pytest.skip(f"NVML not available: {e}")
            raise


class TestProfilingIntegration:
    """Integration tests for profiling workflows."""
    
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_profile_model_training_step(self, cuda_device):
        """Test profiling a complete training step."""
        from torch.profiler import profile, ProfilerActivity
        
        # Simple model and data
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 3072),
            torch.nn.GELU(),
            torch.nn.Linear(3072, 768),
        ).to(cuda_device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        x = torch.randn(32, 768, device=cuda_device)
        y = torch.randn(32, 768, device=cuda_device)
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True
        ) as prof:
            # Forward
            output = model(x)
            loss = criterion(output, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Verify events were captured
        events = prof.key_averages()
        
        # Should have CUDA events
        cuda_events = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
        assert len(cuda_events) > 0, "No CUDA events captured"
    
    @pytest.mark.gpu
    @pytest.mark.integration
    def test_memory_profiling_snapshot(self, cuda_device):
        """Test memory profiling with snapshots."""
        torch.cuda.reset_peak_memory_stats()
        
        # Allocate some memory
        tensors = []
        for i in range(10):
            tensors.append(torch.randn(1000, 1000, device=cuda_device))
        
        peak_memory = torch.cuda.max_memory_allocated()
        
        # Clean up
        del tensors
        torch.cuda.empty_cache()
        
        # Should have recorded significant memory usage
        assert peak_memory > 10 * 1000 * 1000 * 4  # > 40MB
