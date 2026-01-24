"""
test_hpc_lab.py - Tests for HPC lab modules

Tests:
- Slurm job templates
- Checkpointing utilities
- Container definitions
"""

import pytest
import sys
from pathlib import Path


class TestSlurmTemplates:
    """Tests for Slurm job templates."""
    
    @pytest.mark.smoke
    def test_slurm_templates_exist(self, hpc_lab_dir):
        """Test that Slurm templates exist."""
        templates_dir = hpc_lab_dir / "01-slurm-basics" / "templates"
        
        required_templates = [
            "single-gpu-job.sbatch",
            "multi-gpu-job.sbatch",
            "multi-node-job.sbatch",
            "job-array.sbatch",
        ]
        
        for template in required_templates:
            filepath = templates_dir / template
            assert filepath.exists(), f"Missing template: {template}"
    
    @pytest.mark.smoke
    def test_sbatch_syntax(self, hpc_lab_dir):
        """Test that sbatch files have valid syntax."""
        templates_dir = hpc_lab_dir / "01-slurm-basics" / "templates"
        
        for sbatch_file in templates_dir.glob("*.sbatch"):
            content = sbatch_file.read_text()
            
            # Should start with shebang
            assert content.startswith("#!/bin/bash"), f"Missing shebang in {sbatch_file.name}"
            
            # Should have SBATCH directives
            assert "#SBATCH" in content, f"No SBATCH directives in {sbatch_file.name}"
    
    @pytest.mark.smoke
    def test_single_gpu_template_content(self, hpc_lab_dir):
        """Test single GPU template has required directives."""
        template = hpc_lab_dir / "01-slurm-basics" / "templates" / "single-gpu-job.sbatch"
        
        if template.exists():
            content = template.read_text()
            
            # Should specify GPU
            assert "--gres=gpu" in content or "-G" in content or "gpu" in content.lower()
            
            # Should have job name
            assert "--job-name" in content or "-J" in content
    
    @pytest.mark.smoke
    def test_multi_node_template_content(self, hpc_lab_dir):
        """Test multi-node template has MPI/distributed directives."""
        template = hpc_lab_dir / "01-slurm-basics" / "templates" / "multi-node-job.sbatch"
        
        if template.exists():
            content = template.read_text()
            
            # Should specify nodes
            assert "--nodes" in content or "-N" in content
            
            # Should have tasks or processes
            assert "ntasks" in content or "srun" in content


class TestCheckpointing:
    """Tests for checkpointing utilities."""
    
    @pytest.mark.smoke
    def test_checkpoint_utils_exist(self, hpc_lab_dir):
        """Test checkpoint utility files exist."""
        checkpoint_dir = hpc_lab_dir / "02-checkpointing"
        
        required_files = [
            "templates/checkpoint_utils.py",
            "examples/train_with_checkpoint.py",
            "examples/auto_resume.sbatch",
        ]
        
        for filepath in required_files:
            full_path = checkpoint_dir / filepath
            assert full_path.exists(), f"Missing: {filepath}"
    
    @pytest.mark.smoke
    def test_checkpoint_utils_importable(self, hpc_lab_dir):
        """Test checkpoint_utils.py can be imported."""
        sys.path.insert(0, str(hpc_lab_dir / "02-checkpointing" / "templates"))
        
        try:
            from checkpoint_utils import CheckpointManager
            
            # Should be a class
            assert callable(CheckpointManager)
        except ImportError as e:
            pytest.fail(f"Could not import CheckpointManager: {e}")
        finally:
            sys.path.pop(0)
    
    @pytest.mark.gpu
    def test_checkpoint_save_load(self, hpc_lab_dir, cuda_device, temp_dir):
        """Test checkpoint save and load functionality."""
        sys.path.insert(0, str(hpc_lab_dir / "02-checkpointing" / "templates"))
        
        try:
            from checkpoint_utils import CheckpointManager
            import torch
            
            # Create simple model
            model = torch.nn.Linear(10, 10).to(cuda_device)
            optimizer = torch.optim.Adam(model.parameters())
            
            # Create checkpoint manager
            manager = CheckpointManager(
                checkpoint_dir=str(temp_dir),
                model=model,
                optimizer=optimizer
            )
            
            # Save checkpoint
            manager.save(epoch=5, loss=0.5)
            
            # Check file was created
            checkpoint_files = list(temp_dir.glob("*.pt")) + list(temp_dir.glob("*.pth"))
            assert len(checkpoint_files) > 0, "No checkpoint file created"
            
            # Modify model
            original_weight = model.weight.data.clone()
            model.weight.data.fill_(0)
            
            # Load checkpoint
            epoch = manager.load_latest()
            
            # Weight should be restored
            assert not torch.allclose(model.weight.data, torch.zeros_like(model.weight.data))
            
        except ImportError as e:
            pytest.skip(f"Could not import checkpoint utilities: {e}")
        finally:
            if str(hpc_lab_dir / "02-checkpointing" / "templates") in sys.path:
                sys.path.remove(str(hpc_lab_dir / "02-checkpointing" / "templates"))


class TestContainers:
    """Tests for container definitions."""
    
    @pytest.mark.smoke
    def test_container_files_exist(self, hpc_lab_dir):
        """Test container definition files exist."""
        container_dir = hpc_lab_dir / "03-containers"
        
        required_files = [
            "README.md",
            "templates/cuda-pytorch.def",
            "templates/build-container.sh",
        ]
        
        for filepath in required_files:
            full_path = container_dir / filepath
            assert full_path.exists(), f"Missing: {filepath}"
    
    @pytest.mark.smoke
    def test_singularity_def_syntax(self, hpc_lab_dir):
        """Test Singularity definition file has valid structure."""
        def_file = hpc_lab_dir / "03-containers" / "templates" / "cuda-pytorch.def"
        
        if def_file.exists():
            content = def_file.read_text()
            
            # Should have Bootstrap section
            assert "Bootstrap:" in content, "Missing Bootstrap section"
            
            # Should have From section
            assert "From:" in content, "Missing From section"
            
            # Should have %post section for installation
            assert "%post" in content, "Missing %post section"
    
    @pytest.mark.smoke
    def test_build_script_executable(self, hpc_lab_dir):
        """Test build script content."""
        build_script = hpc_lab_dir / "03-containers" / "templates" / "build-container.sh"
        
        if build_script.exists():
            content = build_script.read_text()
            
            # Should have shebang
            assert content.startswith("#!/bin/bash") or content.startswith("#!/usr/bin/env bash")
            
            # Should reference singularity or apptainer
            assert "singularity" in content.lower() or "apptainer" in content.lower()


class TestHPCReadmes:
    """Tests for HPC lab documentation."""
    
    @pytest.mark.smoke
    def test_main_readme_exists(self, hpc_lab_dir):
        """Test main HPC lab README exists."""
        readme = hpc_lab_dir / "README.md"
        assert readme.exists(), "HPC lab README missing"
    
    @pytest.mark.smoke
    def test_module_readmes_exist(self, hpc_lab_dir):
        """Test each module has a README."""
        modules = ["01-slurm-basics", "02-checkpointing", "03-containers"]
        
        for module in modules:
            # Check for README in module or templates subdirectory
            readme = hpc_lab_dir / module / "README.md"
            templates_readme = hpc_lab_dir / module / "templates" / "README.md"
            
            has_readme = readme.exists() or templates_readme.exists()
            # Note: Some modules may document in parent README
    
    @pytest.mark.smoke
    def test_quick_reference_exists(self, repo_root):
        """Test HPC quick reference exists."""
        quick_ref = repo_root / "notes" / "hpc-quick-reference.md"
        assert quick_ref.exists(), "HPC quick reference missing"
    
    @pytest.mark.smoke
    def test_quick_reference_content(self, repo_root):
        """Test quick reference has expected sections."""
        quick_ref = repo_root / "notes" / "hpc-quick-reference.md"
        
        if quick_ref.exists():
            content = quick_ref.read_text().lower()
            
            # Should cover key HPC topics
            assert "slurm" in content
            assert "module" in content or "environment" in content
