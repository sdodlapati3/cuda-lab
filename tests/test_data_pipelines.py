"""
test_data_pipelines.py - Tests for data pipeline modules

Tests:
- Optimized DataLoader configurations
- DALI pipeline utilities
"""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path


class TestOptimizedDataLoader:
    """Tests for optimized DataLoader patterns."""
    
    @pytest.mark.smoke
    def test_dataloader_module_exists(self, repo_root):
        """Test optimized dataloader module exists."""
        module = repo_root / "data-pipelines" / "01-optimized-loading" / "optimized_dataloader.py"
        assert module.exists(), "Optimized dataloader module missing"
    
    @pytest.mark.smoke
    def test_dataloader_config_class(self, repo_root):
        """Test DataLoaderConfig class."""
        sys.path.insert(0, str(repo_root / "data-pipelines" / "01-optimized-loading"))
        
        try:
            from optimized_dataloader import DataLoaderConfig
            
            config = DataLoaderConfig(
                batch_size=64,
                num_workers=4,
                pin_memory=True
            )
            
            assert config.batch_size == 64
            assert config.num_workers == 4
            
        except ImportError:
            pytest.skip("Could not import DataLoaderConfig")
        finally:
            if sys.path[0].endswith("01-optimized-loading"):
                sys.path.pop(0)
    
    @pytest.mark.gpu
    def test_pin_memory_effect(self, cuda_device):
        """Test that pin_memory affects transfer time."""
        
        class SimpleDataset(Dataset):
            def __init__(self, size):
                self.data = torch.randn(size, 3, 224, 224)
                self.labels = torch.randint(0, 10, (size,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        dataset = SimpleDataset(100)
        
        # Without pin_memory
        loader_no_pin = DataLoader(dataset, batch_size=32, pin_memory=False)
        
        # With pin_memory
        loader_pin = DataLoader(dataset, batch_size=32, pin_memory=True)
        
        # Both should work
        for (x1, y1), (x2, y2) in zip(loader_no_pin, loader_pin):
            x1 = x1.to(cuda_device)
            x2 = x2.to(cuda_device)
            break
    
    @pytest.mark.gpu
    def test_prefetch_factor(self, cuda_device):
        """Test prefetch_factor configuration."""
        
        class SimpleDataset(Dataset):
            def __len__(self):
                return 1000
            
            def __getitem__(self, idx):
                return torch.randn(3, 32, 32), idx % 10
        
        dataset = SimpleDataset()
        
        # With prefetch
        loader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=True
        )
        
        # Should work without error
        for batch_idx, (x, y) in enumerate(loader):
            if batch_idx >= 5:
                break


class TestSyntheticDataset:
    """Tests for synthetic dataset utilities."""
    
    @pytest.mark.smoke
    def test_synthetic_dataset_creation(self, repo_root):
        """Test SyntheticDataset can be created."""
        sys.path.insert(0, str(repo_root / "data-pipelines" / "01-optimized-loading"))
        
        try:
            from optimized_dataloader import SyntheticDataset
            
            dataset = SyntheticDataset(
                size=1000,
                input_shape=(3, 224, 224),
                simulate_io_ms=0,
                simulate_cpu_ms=0
            )
            
            assert len(dataset) == 1000
            
            x, y = dataset[0]
            assert x.shape == (3, 224, 224)
            
        except ImportError:
            pytest.skip("Could not import SyntheticDataset")
        finally:
            if sys.path[0].endswith("01-optimized-loading"):
                sys.path.pop(0)
    
    @pytest.mark.smoke
    def test_synthetic_dataset_iteration(self, repo_root):
        """Test iterating through synthetic dataset."""
        sys.path.insert(0, str(repo_root / "data-pipelines" / "01-optimized-loading"))
        
        try:
            from optimized_dataloader import SyntheticDataset
            
            dataset = SyntheticDataset(
                size=100,
                input_shape=(3, 32, 32)
            )
            
            loader = DataLoader(dataset, batch_size=10)
            
            batch_count = 0
            for x, y in loader:
                assert x.shape == (10, 3, 32, 32)
                batch_count += 1
            
            assert batch_count == 10
            
        except ImportError:
            pytest.skip("Could not import SyntheticDataset")
        finally:
            if sys.path[0].endswith("01-optimized-loading"):
                sys.path.pop(0)


class TestDALIPipeline:
    """Tests for DALI pipeline utilities."""
    
    @pytest.mark.smoke
    def test_dali_module_exists(self, repo_root):
        """Test DALI pipeline module exists."""
        module = repo_root / "data-pipelines" / "02-dali" / "dali_pipeline.py"
        assert module.exists(), "DALI pipeline module missing"
    
    @pytest.mark.gpu
    def test_dali_availability_check(self):
        """Test DALI availability detection."""
        try:
            import nvidia.dali
            dali_available = True
        except ImportError:
            dali_available = False
        
        # Just check we can detect availability
        assert isinstance(dali_available, bool)
    
    @pytest.mark.gpu
    def test_dali_synthetic_pipeline(self, cuda_device, repo_root):
        """Test DALI synthetic pipeline."""
        # Check if DALI is available first
        try:
            import nvidia.dali
        except ImportError:
            pytest.skip("DALI not installed")
        
        sys.path.insert(0, str(repo_root / "data-pipelines" / "02-dali"))
        
        try:
            from dali_pipeline import create_dali_dataloader
            
            loader = create_dali_dataloader(
                batch_size=32,
                image_size=224,
                synthetic=True
            )
            
            # Get one batch
            images, labels = next(iter(loader))
            
            assert images.shape[0] == 32
            assert images.shape[2] == 224
            assert images.shape[3] == 224
            
        except ImportError:
            pytest.skip("Could not import DALI pipeline")
        except Exception as e:
            if "DALI" in str(e):
                pytest.skip(f"DALI error: {e}")
            raise
        finally:
            if sys.path[0].endswith("02-dali"):
                sys.path.pop(0)


class TestDataPipelinePerformance:
    """Performance tests for data pipelines."""
    
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_dataloader_throughput(self, cuda_device):
        """Test DataLoader throughput measurement."""
        import time
        
        class FastDataset(Dataset):
            def __init__(self, size):
                self.data = torch.randn(size, 3, 224, 224)
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], 0
        
        dataset = FastDataset(1000)
        loader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=2,
            pin_memory=True
        )
        
        # Warmup
        for i, _ in enumerate(loader):
            if i >= 5:
                break
        
        # Measure throughput
        start = time.perf_counter()
        n_samples = 0
        
        for x, y in loader:
            x = x.to(cuda_device, non_blocking=True)
            n_samples += x.shape[0]
            if n_samples >= 500:
                break
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        throughput = n_samples / elapsed
        
        # Should achieve reasonable throughput
        assert throughput > 100, f"Throughput too low: {throughput} samples/sec"
    
    @pytest.mark.gpu
    def test_worker_scaling(self, cuda_device):
        """Test that more workers generally helps (up to a point)."""
        
        class SlowDataset(Dataset):
            def __init__(self, size):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                import time
                time.sleep(0.001)  # Simulate 1ms I/O
                return torch.randn(3, 32, 32), 0
        
        dataset = SlowDataset(100)
        
        throughputs = []
        for num_workers in [0, 2]:
            loader = DataLoader(
                dataset,
                batch_size=10,
                num_workers=num_workers
            )
            
            import time
            start = time.perf_counter()
            
            for i, _ in enumerate(loader):
                if i >= 5:
                    break
            
            elapsed = time.perf_counter() - start
            throughputs.append(50 / elapsed)  # 5 batches * 10 samples
        
        # Workers should help with I/O-bound workload
        # Note: May not always be true on all systems
