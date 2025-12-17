#!/bin/bash
# Setup CUDA 13.1 environment for L40/B200/H100 GPUs
# This uses the new numba-cuda package (separate from numba)

set -e

ENV_PATH="${1:-$HOME/envs/cuda13}"

echo "=========================================="
echo "Setting up CUDA 13.1 Environment"
echo "=========================================="
echo "Environment path: $ENV_PATH"
echo ""
echo "Target GPUs: B200, H100, A100, L40"
echo ""

# Load Python module
echo "Loading Python 3.12 module..."
module load python3

# Create environment if it doesn't exist
if [ -d "$ENV_PATH" ]; then
    echo "✅ Environment already exists at $ENV_PATH"
    echo "   To recreate, run: rm -rf $ENV_PATH"
else
    echo "Creating new Python 3.12 environment..."
    crun -c -p "$ENV_PATH"
    echo "✅ Environment created"
fi

# Install packages
echo ""
echo "Installing CUDA 13.1 compatible packages..."
crun -p "$ENV_PATH" pip install --quiet --upgrade pip

# Core packages
crun -p "$ENV_PATH" pip install --quiet numpy matplotlib jupyter jupyterlab ipywidgets

# CUDA packages - NEW numba-cuda is separate!
echo "Installing numba + numba-cuda (CUDA 13 support)..."
crun -p "$ENV_PATH" pip install --quiet numba numba-cuda

# NVIDIA CUDA Python bindings (recommended for CUDA 13)
echo "Installing NVIDIA CUDA Python bindings..."
crun -p "$ENV_PATH" pip install --quiet cuda-python

# Optional: cupy for GPU arrays (alternative to numba for some use cases)
echo "Installing CuPy (GPU NumPy)..."
crun -p "$ENV_PATH" pip install --quiet cupy-cuda12x || echo "CuPy install skipped (may need CUDA 13 version)"

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
crun -p "$ENV_PATH" python -c "
import sys
print('Python:', sys.version.split()[0])

import numpy as np
print('NumPy:', np.__version__)

import numba
print('Numba:', numba.__version__)

try:
    import numba_cuda
    print('numba-cuda: installed ✅')
except ImportError:
    print('numba-cuda: not found (using built-in)')

try:
    from cuda import cuda as cuda_driver
    print('cuda-python: installed ✅')
except ImportError:
    print('cuda-python: not found')

from numba import cuda
print('CUDA available:', cuda.is_available())
if cuda.is_available():
    dev = cuda.get_current_device()
    print(f'  GPU: {dev.name.decode()}')
    print(f'  Compute Capability: {dev.compute_capability}')
"

echo ""
echo "=========================================="
echo "✅ CUDA 13.1 Environment Ready!"
echo "=========================================="
echo ""
echo "Available GPU partitions on this cluster:"
echo "  • b200flex    - 1 node × 8 B200 GPUs (Blackwell)"
echo "  • h100octflex - 2 nodes × 8 H100 GPUs"
echo "  • h100quadflex - 4 nodes × 4 H100 GPUs"
echo "  • h100flex    - 15 nodes × 1 H100 GPU"
echo "  • a100flex    - 10 nodes × 1 A100 GPU"
echo ""
echo "Quick start commands:"
echo "  # Request B200 GPU (8 GPUs per node!)"
echo "  srun --partition=b200flex --gres=gpu:1 -c 8 --time=01:00:00 --pty bash"
echo ""
echo "  # Request H100 GPU"
echo "  srun --partition=h100flex --gres=gpu:1 -c 4 --time=04:00:00 --pty bash"
echo ""
echo "  # Then run Python:"
echo "  module load python3"
echo "  crun -p $ENV_PATH python your_script.py"
