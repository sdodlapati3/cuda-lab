# CUDA Lab Test Suite

Comprehensive tests for validating all modules in the CUDA Lab repository.

## Overview

This test suite validates:
- Profiling lab exercises and tools
- HPC lab templates and utilities
- Scientific ML implementations
- Benchmark infrastructure
- Data pipeline utilities
- PyTorch optimization examples

## Running Tests

```bash
# Load environment
module load python3

# Install test dependencies (if needed)
crun -p ~/envs/cuda-lab pip install pytest pytest-cov pytest-timeout

# Run all tests
crun -p ~/envs/cuda-lab pytest tests/ -v

# Run with coverage
crun -p ~/envs/cuda-lab pytest tests/ --cov=. --cov-report=html

# Run specific module tests
crun -p ~/envs/cuda-lab pytest tests/test_benchmarks.py -v
crun -p ~/envs/cuda-lab pytest tests/test_profiling.py -v

# Run quick smoke tests only
crun -p ~/envs/cuda-lab pytest tests/ -v -m "smoke"

# Run GPU tests only (requires CUDA)
crun -p ~/envs/cuda-lab pytest tests/ -v -m "gpu"
```

## Test Categories

| Category | Marker | Description |
|----------|--------|-------------|
| Smoke tests | `@pytest.mark.smoke` | Quick validation tests |
| GPU tests | `@pytest.mark.gpu` | Requires CUDA GPU |
| Slow tests | `@pytest.mark.slow` | Long-running tests |
| Integration | `@pytest.mark.integration` | End-to-end tests |

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_benchmarks.py       # Benchmark module tests
├── test_profiling.py        # Profiling lab tests
├── test_hpc_lab.py          # HPC utilities tests
├── test_scientific_ml.py    # Scientific ML tests
├── test_data_pipelines.py   # Data loading tests
├── test_pytorch_opt.py      # PyTorch optimization tests
└── test_integration.py      # End-to-end tests
```

## Writing New Tests

Follow these guidelines:
1. Use descriptive test names: `test_<module>_<functionality>_<scenario>`
2. Add appropriate markers for categorization
3. Use fixtures for common setup
4. Include docstrings explaining what's being tested
