#!/bin/bash
# Setup CUDA learning environment (run once)

echo "=========================================="
echo "Setting up CUDA Learning Environment"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "✅ Miniconda installed"
else
    echo "✅ Conda already available"
fi

# Create environment
echo ""
echo "Creating cuda-learning environment..."
conda create -n cuda-learning python=3.10 -y

echo ""
echo "Installing packages..."
conda activate cuda-learning
pip install numba numpy jupyter jupyterlab matplotlib

# Verify
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "
from numba import cuda
import numpy as np
print('✅ NumPy:', np.__version__)
print('✅ Numba installed')
print('✅ CUDA module available:', hasattr(cuda, 'jit'))
"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "To start learning:"
echo "  1. Run: ./scripts/gpu-session.sh"
echo "  2. Or submit: sbatch scripts/start-jupyter-gpu.sh"
echo ""
