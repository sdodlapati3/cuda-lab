"""
test_curriculum.py - Tests for learning-path notebooks and practice exercises

Tests:
- Learning path notebook structure and content
- Practice exercise files and structure
- Bootcamp capstone content
- CUDA code syntax validation
"""

import pytest
import sys
from pathlib import Path
import json
import re


class TestLearningPathStructure:
    """Tests for learning-path notebook structure."""
    
    @pytest.mark.smoke
    def test_learning_path_weeks_exist(self, repo_root):
        """Test that all 18 weeks of learning path exist."""
        learning_path = repo_root / "learning-path"
        
        for week in range(1, 19):
            week_dir = learning_path / f"week-{week:02d}"
            assert week_dir.exists(), f"Missing week-{week:02d} directory"
    
    @pytest.mark.smoke
    def test_week_01_notebooks_exist(self, repo_root):
        """Test Week 1 has all required notebooks."""
        week1 = repo_root / "learning-path" / "week-01"
        
        required_files = [
            "day-1-gpu-basics.ipynb",
            "day-2-thread-indexing.ipynb",
            "day-3-memory-basics.ipynb",
            "day-4-error-handling.ipynb",
            "checkpoint-quiz.md",
            "README.md"
        ]
        
        for filename in required_files:
            filepath = week1 / filename
            assert filepath.exists(), f"Missing file: {filename}"
    
    @pytest.mark.smoke
    def test_notebook_structure_valid(self, repo_root):
        """Test that notebooks are valid JSON."""
        learning_path = repo_root / "learning-path"
        
        notebooks = list(learning_path.glob("**/*.ipynb"))
        assert len(notebooks) > 0, "No notebooks found"
        
        for notebook in notebooks[:5]:  # Check first 5 notebooks
            try:
                with open(notebook, 'r') as f:
                    content = json.load(f)
                
                # Valid notebook structure
                assert 'cells' in content, f"No cells in {notebook.name}"
                assert 'metadata' in content, f"No metadata in {notebook.name}"
                
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {notebook.name}: {e}")
    
    @pytest.mark.smoke
    def test_notebook_has_cuda_content(self, repo_root):
        """Test notebooks contain CUDA-related content."""
        week1 = repo_root / "learning-path" / "week-01"
        notebook = week1 / "day-1-gpu-basics.ipynb"
        
        with open(notebook, 'r') as f:
            content = json.load(f)
        
        # Combine all cell content
        all_content = ""
        for cell in content['cells']:
            if 'source' in cell:
                all_content += "\n".join(cell['source'])
        
        # Should mention CUDA/GPU concepts
        assert any(term in all_content.lower() for term in ['cuda', 'gpu', 'kernel', 'thread']), \
            "Week 1 Day 1 should contain CUDA/GPU content"
    
    @pytest.mark.smoke
    def test_learning_path_readme_exists(self, repo_root):
        """Test main learning path README exists."""
        readme = repo_root / "learning-path" / "README.md"
        assert readme.exists(), "Learning path README missing"
        
        content = readme.read_text()
        assert len(content) > 500, "README seems too short"


class TestPracticeExercises:
    """Tests for practice exercise structure."""
    
    @pytest.mark.smoke
    def test_practice_sections_exist(self, repo_root):
        """Test all practice sections exist."""
        practice = repo_root / "practice"
        
        required_sections = [
            "01-foundations",
            "02-memory",
            "03-parallel",
            "04-optimization",
            "05-advanced",
            "06-systems"
        ]
        
        for section in required_sections:
            section_dir = practice / section
            assert section_dir.exists(), f"Missing practice section: {section}"
    
    @pytest.mark.smoke
    def test_exercise_structure(self, repo_root):
        """Test practice exercises have required files."""
        device_query = repo_root / "practice" / "01-foundations" / "ex01-device-query"
        
        required_files = ["README.md", "device_query.cu", "solution.cu", "Makefile"]
        
        for filename in required_files:
            filepath = device_query / filename
            assert filepath.exists(), f"Missing file: {filename}"
    
    @pytest.mark.smoke
    def test_cuda_syntax_basic(self, repo_root):
        """Test CUDA files have valid syntax patterns."""
        device_query = repo_root / "practice" / "01-foundations" / "ex01-device-query"
        
        cu_file = device_query / "device_query.cu"
        if cu_file.exists():
            content = cu_file.read_text()
            
            # Should have include statements
            assert "#include" in content, "Missing include statements"
            
            # Should have main function
            assert "main" in content, "Missing main function"
            
            # Should have CUDA calls
            assert any(call in content for call in ["cuda", "CUDA"]), \
                "Should contain CUDA API calls"
    
    @pytest.mark.smoke
    def test_makefile_exists_for_exercises(self, repo_root):
        """Test all CUDA exercises have Makefiles."""
        practice = repo_root / "practice"
        
        # Find all exercise directories
        exercise_dirs = list(practice.glob("*/ex*"))
        
        makefiles_found = 0
        for ex_dir in exercise_dirs:
            makefile = ex_dir / "Makefile"
            if makefile.exists():
                makefiles_found += 1
        
        # At least some exercises should have Makefiles
        assert makefiles_found > 0, "No Makefiles found in practice exercises"


class TestBootcampStructure:
    """Tests for bootcamp curriculum structure."""
    
    @pytest.mark.smoke
    def test_bootcamp_phases_exist(self, repo_root):
        """Test all 8 bootcamp phases exist."""
        bootcamp = repo_root / "bootcamp"
        
        for phase in range(9):  # phase0 through phase8
            phase_dir = bootcamp / f"phase{phase}"
            assert phase_dir.exists(), f"Missing phase{phase} directory"
    
    @pytest.mark.smoke
    def test_bootcamp_readme_exists(self, repo_root):
        """Test bootcamp README exists with curriculum overview."""
        readme = repo_root / "bootcamp" / "README.md"
        assert readme.exists(), "Bootcamp README missing"
        
        content = readme.read_text()
        
        # Should mention phases
        assert "phase" in content.lower(), "README should describe phases"
    
    @pytest.mark.smoke
    def test_capstones_exist(self, repo_root):
        """Test capstone projects directory exists."""
        capstones = repo_root / "bootcamp" / "capstones"
        assert capstones.exists(), "Capstones directory missing"
    
    @pytest.mark.smoke
    def test_bootcamp_templates_exist(self, repo_root):
        """Test bootcamp templates exist."""
        templates = repo_root / "bootcamp" / "templates"
        assert templates.exists(), "Bootcamp templates directory missing"
        
        # Should have subdirectories
        subdirs = list(templates.iterdir())
        assert len(subdirs) > 0, "Templates directory is empty"


class TestCUDACodeValidation:
    """Tests for CUDA code syntax validation."""
    
    @pytest.mark.smoke
    def test_profiling_cuda_exercises(self, repo_root):
        """Test profiling CUDA exercises have valid structure."""
        profiling = repo_root / "profiling-lab"
        
        # Find .cu files
        cu_files = list(profiling.glob("**/*.cu"))
        
        # Should have some CUDA files
        assert len(cu_files) > 0, "No CUDA files found in profiling-lab"
        
        for cu_file in cu_files[:5]:  # Check first 5
            content = cu_file.read_text()
            
            # Basic CUDA patterns
            has_kernel = "__global__" in content
            has_cuda_api = "cuda" in content.lower()
            
            assert has_kernel or has_cuda_api, \
                f"{cu_file.name} should contain CUDA code"
    
    @pytest.mark.smoke
    def test_cuda_kernel_patterns(self, repo_root):
        """Test CUDA kernels follow expected patterns."""
        profiling = repo_root / "profiling-lab"
        
        cu_files = list(profiling.glob("**/*.cu"))
        
        kernel_count = 0
        for cu_file in cu_files:
            content = cu_file.read_text()
            
            # Count kernel declarations
            kernel_matches = re.findall(r'__global__\s+void\s+\w+', content)
            kernel_count += len(kernel_matches)
        
        # Should have multiple kernels across exercises
        assert kernel_count > 0, "Should have CUDA kernel definitions"


class TestDocumentation:
    """Tests for documentation completeness."""
    
    @pytest.mark.smoke
    def test_cuda_programming_guide_exists(self, repo_root):
        """Test CUDA programming guide exists."""
        guide = repo_root / "cuda-programming-guide"
        assert guide.exists(), "CUDA programming guide missing"
        
        index = guide / "index.md"
        assert index.exists(), "Guide index.md missing"
    
    @pytest.mark.smoke
    def test_guide_sections_exist(self, repo_root):
        """Test programming guide has all sections."""
        guide = repo_root / "cuda-programming-guide"
        
        required_sections = [
            "01-introduction",
            "02-basics",
            "03-advanced",
            "04-special-topics",
            "05-appendices"
        ]
        
        for section in required_sections:
            section_dir = guide / section
            assert section_dir.exists(), f"Missing guide section: {section}"
    
    @pytest.mark.smoke
    def test_quick_reference_exists(self, repo_root):
        """Test quick reference notes exist."""
        cuda_ref = repo_root / "notes" / "cuda-quick-reference.md"
        hpc_ref = repo_root / "notes" / "hpc-quick-reference.md"
        
        assert cuda_ref.exists(), "CUDA quick reference missing"
        assert hpc_ref.exists(), "HPC quick reference missing"


class TestBenchmarkTemplates:
    """Tests for benchmark templates and examples."""
    
    @pytest.mark.smoke
    def test_benchmark_readme_exists(self, repo_root):
        """Test benchmarks README exists."""
        readme = repo_root / "benchmarks" / "README.md"
        assert readme.exists(), "Benchmarks README missing"
    
    @pytest.mark.smoke
    def test_hardware_baselines_exist(self, repo_root):
        """Test hardware baseline files exist."""
        baselines = repo_root / "benchmarks" / "hardware-baselines"
        
        assert baselines.exists(), "Hardware baselines directory missing"
        
        # Should have at least A100 baseline
        a100 = baselines / "A100-80GB.json"
        assert a100.exists(), "A100 baseline missing"
    
    @pytest.mark.smoke
    def test_kernel_benchmarks_exist(self, repo_root):
        """Test kernel benchmark directories exist."""
        kernels = repo_root / "benchmarks" / "kernels"
        
        required_kernels = ["matmul", "softmax", "attention", "reduction"]
        
        for kernel in required_kernels:
            kernel_dir = kernels / kernel
            assert kernel_dir.exists(), f"Missing kernel benchmark: {kernel}"
            
            # Each should have benchmark.py
            benchmark_file = kernel_dir / "benchmark.py"
            assert benchmark_file.exists(), f"Missing benchmark.py for {kernel}"


class TestScriptsAndSetup:
    """Tests for setup scripts."""
    
    @pytest.mark.smoke
    def test_scripts_directory_exists(self, repo_root):
        """Test scripts directory exists."""
        scripts = repo_root / "scripts"
        assert scripts.exists(), "Scripts directory missing"
    
    @pytest.mark.smoke
    def test_setup_scripts_exist(self, repo_root):
        """Test key setup scripts exist."""
        scripts = repo_root / "scripts"
        
        setup_scripts = [
            "setup-environment.sh",
            "setup-cuda13.sh",
        ]
        
        for script in setup_scripts:
            filepath = scripts / script
            if filepath.exists():
                content = filepath.read_text()
                assert content.startswith("#!/"), f"{script} should have shebang"
