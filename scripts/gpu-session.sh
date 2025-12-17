#!/bin/bash
# Quick interactive GPU session for CUDA learning
# Usage: ./gpu-session.sh

echo "🚀 Requesting T4 GPU node..."
echo "   Partition: t4flex"
echo "   GPU: 1x T4"
echo "   Time: 4 hours"
echo ""

srun --partition=t4flex \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --mem=16G \
     --time=04:00:00 \
     --pty bash -c '
echo "=========================================="
echo "✅ GPU Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=========================================="
echo ""
echo "Activating CUDA environment..."
source ~/miniconda3/bin/activate cuda-learning 2>/dev/null || conda activate cuda-learning
echo ""
echo "🎓 Ready! You can now:"
echo "   1. Run: python verify_cuda.py"
echo "   2. Run: jupyter notebook --no-browser --port=8888"
echo "   3. Run: python -c \"from numba import cuda; print(cuda.get_current_device().name.decode())\""
echo ""
cd ~/cuda-lab
exec bash
'
