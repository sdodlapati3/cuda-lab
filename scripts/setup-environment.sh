#!/bin/bash
# Setup CUDA learning environment for ODU HPC cluster
# Uses Python 3.12 module with crun container system

set -e

ENV_PATH="${1:-$HOME/envs/cuda_lab}"

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
crun -p "$ENV_PATH" pip install --quiet numba numpy jupyter jupyterlab matplotlib ipywidgets

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
echo "  Run Python:  crun -p $ENV_PATH python script.py"
echo "  Run Jupyter: crun -p $ENV_PATH jupyter lab"
echo ""
echo "To start learning on GPU:"
echo "  1. Interactive: ./scripts/gpu-session.sh"
echo "  2. Jupyter job: sbatch scripts/start-jupyter-gpu.sh"
echo ""
