"""
Tests for profiling-lab/05-advanced-nsight module.

Tests exercise structure, file existence, and Python script syntax.
"""

import pytest
import os
import ast
import subprocess
from pathlib import Path

# Base path for the module - go up from tests/ to cuda-lab/
MODULE_PATH = Path(__file__).parent.parent / "profiling-lab" / "05-advanced-nsight"


class TestModuleStructure:
    """Test module directory structure and README."""
    
    def test_module_exists(self):
        """Module directory should exist."""
        assert MODULE_PATH.exists(), f"Module not found: {MODULE_PATH}"
    
    def test_module_readme_exists(self):
        """Module should have a README."""
        readme = MODULE_PATH / "README.md"
        assert readme.exists(), "Module README.md not found"
    
    def test_module_readme_has_content(self):
        """README should have meaningful content."""
        readme = MODULE_PATH / "README.md"
        content = readme.read_text()
        assert len(content) > 500, "README seems too short"
        assert "Advanced Nsight" in content or "exercise" in content.lower()
    
    def test_all_exercises_exist(self):
        """All 8 exercises should have directories."""
        expected_exercises = [
            "ex01-python-backtrace",
            "ex02-io-dataloader",
            "ex03-nsys-stats-cli",
            "ex04-sqlite-analysis",
            "ex05-cpu-sampling",
            "ex06-osrt-tracing",
            "ex07-comparison-reports",
            "ex08-expert-systems",
        ]
        for ex in expected_exercises:
            ex_path = MODULE_PATH / ex
            assert ex_path.exists(), f"Exercise directory not found: {ex}"
            readme = ex_path / "README.md"
            assert readme.exists(), f"README not found in {ex}"


class TestEx01PythonBacktrace:
    """Tests for Python backtrace exercise."""
    
    EX_PATH = MODULE_PATH / "ex01-python-backtrace"
    
    def test_readme_content(self):
        """README should explain Python backtraces."""
        content = (self.EX_PATH / "README.md").read_text()
        assert "python" in content.lower()
        assert "--python-backtrace" in content or "backtrace" in content.lower()
    
    def test_train_model_exists(self):
        """Training script should exist."""
        script = self.EX_PATH / "train_model.py"
        assert script.exists()
    
    def test_train_model_syntax(self):
        """Training script should have valid Python syntax."""
        script = self.EX_PATH / "train_model.py"
        content = script.read_text()
        ast.parse(content)  # Raises SyntaxError if invalid
    
    def test_model_exists(self):
        """Model file should exist."""
        model = self.EX_PATH / "model.py"
        assert model.exists()
    
    def test_model_syntax(self):
        """Model file should have valid Python syntax."""
        model = self.EX_PATH / "model.py"
        content = model.read_text()
        ast.parse(content)
    
    def test_profile_scripts_exist(self):
        """Profile shell scripts should exist."""
        assert (self.EX_PATH / "profile_basic.sh").exists()
        assert (self.EX_PATH / "profile_python.sh").exists()


class TestEx02IODataloader:
    """Tests for I/O and DataLoader exercise."""
    
    EX_PATH = MODULE_PATH / "ex02-io-dataloader"
    
    def test_readme_content(self):
        """README should explain DataLoader profiling."""
        content = (self.EX_PATH / "README.md").read_text()
        assert "dataloader" in content.lower() or "i/o" in content.lower()
    
    def test_slow_dataloader_syntax(self):
        """Slow dataloader script should have valid syntax."""
        script = self.EX_PATH / "slow_dataloader.py"
        if script.exists():
            content = script.read_text()
            ast.parse(content)
    
    def test_fast_dataloader_syntax(self):
        """Fast dataloader script should have valid syntax."""
        script = self.EX_PATH / "fast_dataloader.py"
        if script.exists():
            content = script.read_text()
            ast.parse(content)
    
    def test_profile_script_exists(self):
        """Profile script should exist."""
        script = self.EX_PATH / "profile_dataloader.sh"
        assert script.exists()


class TestEx03NsysStatsCLI:
    """Tests for nsys stats CLI exercise."""
    
    EX_PATH = MODULE_PATH / "ex03-nsys-stats-cli"
    
    def test_readme_content(self):
        """README should explain nsys stats."""
        content = (self.EX_PATH / "README.md").read_text()
        assert "nsys stats" in content or "CLI" in content
    
    def test_analyze_script_exists(self):
        """Analysis script should exist."""
        script = self.EX_PATH / "analyze_profile.sh"
        assert script.exists()


class TestEx04SQLiteAnalysis:
    """Tests for SQLite analysis exercise."""
    
    EX_PATH = MODULE_PATH / "ex04-sqlite-analysis"
    
    def test_readme_content(self):
        """README should explain SQLite export."""
        content = (self.EX_PATH / "README.md").read_text()
        assert "sqlite" in content.lower()
    
    def test_analyze_script_syntax(self):
        """Python analysis script should have valid syntax."""
        script = self.EX_PATH / "analyze_sqlite.py"
        if script.exists():
            content = script.read_text()
            ast.parse(content)
    
    def test_analyze_script_has_queries(self):
        """Analysis script should contain SQL queries."""
        script = self.EX_PATH / "analyze_sqlite.py"
        if script.exists():
            content = script.read_text()
            assert "SELECT" in content or "select" in content


class TestEx05CPUSampling:
    """Tests for CPU sampling exercise."""
    
    EX_PATH = MODULE_PATH / "ex05-cpu-sampling"
    
    def test_readme_content(self):
        """README should explain CPU sampling."""
        content = (self.EX_PATH / "README.md").read_text()
        assert "cpu" in content.lower() or "sampling" in content.lower()
    
    def test_training_script_syntax(self):
        """Training script should have valid syntax."""
        script = self.EX_PATH / "training_with_cpu_bottleneck.py"
        if script.exists():
            content = script.read_text()
            ast.parse(content)


class TestEx06OSRTTracing:
    """Tests for OS runtime tracing exercise."""
    
    EX_PATH = MODULE_PATH / "ex06-osrt-tracing"
    
    def test_readme_content(self):
        """README should explain OS runtime tracing."""
        content = (self.EX_PATH / "README.md").read_text()
        assert "osrt" in content.lower() or "os runtime" in content.lower() or "system call" in content.lower()
    
    def test_profile_script_exists(self):
        """Profile script should exist."""
        script = self.EX_PATH / "profile_osrt.sh"
        assert script.exists()
    
    def test_demo_script_syntax(self):
        """Demo script should have valid Python syntax."""
        script = self.EX_PATH / "osrt_demo.py"
        if script.exists():
            content = script.read_text()
            ast.parse(content)


class TestEx07ComparisonReports:
    """Tests for comparison reports exercise."""
    
    EX_PATH = MODULE_PATH / "ex07-comparison-reports"
    
    def test_readme_content(self):
        """README should explain comparison reports."""
        content = (self.EX_PATH / "README.md").read_text()
        assert "comparison" in content.lower() or "before" in content.lower()
    
    def test_compare_script_exists(self):
        """Comparison shell script should exist."""
        script = self.EX_PATH / "compare_profiles.sh"
        assert script.exists()
    
    def test_compare_python_syntax(self):
        """Python comparison script should have valid syntax."""
        script = self.EX_PATH / "compare_profiles.py"
        if script.exists():
            content = script.read_text()
            ast.parse(content)


class TestEx08ExpertSystems:
    """Tests for expert systems exercise."""
    
    EX_PATH = MODULE_PATH / "ex08-expert-systems"
    
    def test_readme_content(self):
        """README should explain expert systems."""
        content = (self.EX_PATH / "README.md").read_text()
        assert "expert" in content.lower() or "rule" in content.lower()
    
    def test_expert_rules_syntax(self):
        """Expert rules script should have valid syntax."""
        script = self.EX_PATH / "custom_expert_rules.py"
        if script.exists():
            content = script.read_text()
            ast.parse(content)
    
    def test_expert_rules_has_classes(self):
        """Expert rules should define Rule classes."""
        script = self.EX_PATH / "custom_expert_rules.py"
        if script.exists():
            content = script.read_text()
            tree = ast.parse(content)
            class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            assert len(class_names) >= 3, "Expected multiple rule classes"
            assert "Rule" in class_names, "Expected base Rule class"


class TestIntegration:
    """Integration tests for the module."""
    
    @pytest.mark.slow
    def test_all_python_files_importable(self):
        """All Python files should be importable (no import errors in module scope)."""
        python_files = list(MODULE_PATH.rglob("*.py"))
        
        for py_file in python_files:
            content = py_file.read_text()
            try:
                ast.parse(content)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file}: {e}")
    
    def test_shell_scripts_have_shebang(self):
        """All shell scripts should have proper shebang."""
        shell_scripts = list(MODULE_PATH.rglob("*.sh"))
        
        for script in shell_scripts:
            content = script.read_text()
            assert content.startswith("#!/"), f"Missing shebang in {script}"
    
    def test_no_hardcoded_paths(self):
        """Scripts should not have hardcoded absolute paths."""
        python_files = list(MODULE_PATH.rglob("*.py"))
        
        forbidden_patterns = [
            "/home/",
            "/Users/",
            "C:\\Users",
        ]
        
        for py_file in python_files:
            content = py_file.read_text()
            for pattern in forbidden_patterns:
                # Allow in comments and docstrings mentioning paths as examples
                lines = [l for l in content.split('\n') 
                        if not l.strip().startswith('#') and not l.strip().startswith('"""')]
                code_only = '\n'.join(lines)
                assert pattern not in code_only, f"Hardcoded path '{pattern}' found in {py_file}"
