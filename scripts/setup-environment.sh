#!/bin/bash
# Setup CUDA learning environment for ODU HPC cluster
# Uses Python 3.12 module with crun container system
#
# Usage:
#   ./scripts/setup-environment.sh           # Create/verify environment
#   ./scripts/setup-environment.sh --force   # Recreate from scratch

set -e

# Dedicated environment for cuda-lab
ENV_PATH="$HOME/envs/cuda-lab"

# Parse arguments
for arg in "$@"; do
    if [ "$arg" = "--force" ] && [ -d "$ENV_PATH" ]; then
        echo "Removing existing environment..."
        rm -rf "$ENV_PATH"
    fi
done

echo "=========================================="
echo "Setting up CUDA Learning Environment"
echo "=========================================="
echo "Environment path: $ENV_PATH"
echo ""

# Load Python module
echo "Loading Python 3.12 module..."
module load python3

# Create environment if it doesn't exist
if [ -d "$ENV_PATH" ]; then
    echo "✅ Environment already exists at $ENV_PATH"
else
    echo "Creating new Python 3.12 environment..."
    crun -c -p "$ENV_PATH"
    echo "✅ Environment created"
fi

# Install packages
echo ""
echo "Installing/verifying CUDA learning packages..."
crun -p "$ENV_PATH" pip install --quiet --upgrade pip

# Core packages
crun -p "$ENV_PATH" pip install --quiet \
    numpy \
    matplotlib \
    jupyter \
    jupyterlab \
    ipywidgets \
    numba \
    scipy \
    pandas

# ML packages
echo "Installing ML packages..."
crun -p "$ENV_PATH" pip install --quiet torch 2>/dev/null || echo "  PyTorch: skipped (may need manual install)"

# Testing packages
echo "Installing test packages..."
crun -p "$ENV_PATH" pip install --quiet pytest pytest-cov pytest-timeout

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
crun -p "$ENV_PATH" python -c "
import numpy as np
from numba import cuda
print('✅ Python:', __import__('sys').version.split()[0])
print('✅ NumPy:', np.__version__)
print('✅ Numba:', __import__('numba').__version__)
print('✅ CUDA module:', 'available' if hasattr(cuda, 'jit') else 'missing')
print('')
print('Note: CUDA will show as unavailable on login node.')
print('      It will work on GPU nodes (t4flex partition).')
"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Quick reference:"
echo "  Load env:    module load python3"
echo "  Run Python:  crun -p ~/envs/cuda-lab python script.py"
echo "  Run pytest:  crun -p ~/envs/cuda-lab pytest tests/"
echo "  Run Jupyter: crun -p ~/envs/cuda-lab jupyter lab"
echo ""
echo "Alias for convenience (add to ~/.bashrc):"
echo "  alias cudalab='module load python3 && crun -p ~/envs/cuda-lab'"
echo ""
echo "Then use: cudalab python script.py"
echo ""
