#!/bin/bash
# Quick interactive GPU session for CUDA learning
# Usage: ./gpu-session.sh

ENV_PATH="${1:-$HOME/envs/cuda_lab}"

echo "🚀 Requesting T4 GPU node..."
echo "   Partition: t4flex"
echo "   GPU: 1x T4"
echo "   Time: 4 hours"
echo "   Environment: $ENV_PATH"
echo ""

srun --partition=t4flex \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --mem=16G \
     --time=04:00:00 \
     --pty bash -c "
echo '=========================================='
echo '✅ GPU Node:' \$(hostname)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo '=========================================='
echo ''
echo 'Loading Python 3.12 environment...'
module load python3
echo ''
echo '🎓 Ready! Run commands with crun:'
echo '   crun -p $ENV_PATH python ~/cuda-lab/verify_cuda.py'
echo '   crun -p $ENV_PATH jupyter lab --no-browser --port=8888'
echo '   crun -p $ENV_PATH python -c \"from numba import cuda; print(cuda.get_current_device().name.decode())\"'
echo ''
echo 'Or start an interactive Python shell:'
echo '   crun -p $ENV_PATH python'
echo ''
cd ~/cuda-lab
exec bash
"
