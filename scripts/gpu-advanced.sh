#!/bin/bash
# Interactive GPU session for advanced GPUs (B200, H100, A100)
# Usage: ./gpu-advanced.sh [partition] [gpus]
#
# Examples:
#   ./gpu-advanced.sh              # Default: h100flex, 1 GPU
#   ./gpu-advanced.sh b200flex 2   # B200 with 2 GPUs
#   ./gpu-advanced.sh a100flex 1   # A100 with 1 GPU

PARTITION="${1:-h100flex}"
NUM_GPUS="${2:-1}"
ENV_PATH="${CUDA13_ENV:-$HOME/envs/cuda13}"

echo "🚀 Requesting GPU node..."
echo "   Partition: $PARTITION"
echo "   GPUs: $NUM_GPUS"
echo "   Environment: $ENV_PATH"
echo "   Time: 4 hours"
echo ""

# Validate partition
case $PARTITION in
    b200flex)
        echo "   GPU Type: NVIDIA B200 (Blackwell) - 192GB HBM3e"
        ;;
    h100flex|h100dualflex|h100quadflex|h100octflex|h100spot)
        echo "   GPU Type: NVIDIA H100 (Hopper) - 80GB HBM3"
        ;;
    a100flex)
        echo "   GPU Type: NVIDIA A100 (Ampere) - 40/80GB HBM2e"
        ;;
    t4flex)
        echo "   GPU Type: NVIDIA T4 (Turing) - 16GB GDDR6"
        echo "   Note: For T4, use ./gpu-session.sh instead"
        ;;
    *)
        echo "   Unknown partition: $PARTITION"
        echo "   Available: b200flex, h100flex, h100octflex, a100flex, t4flex"
        ;;
esac
echo ""

srun --partition=$PARTITION \
     --gres=gpu:$NUM_GPUS \
     --cpus-per-task=8 \
     --mem=64G \
     --time=04:00:00 \
     --pty bash -c "
echo '=========================================='
echo '✅ GPU Node:' \$(hostname)
echo '=========================================='
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
echo ''
echo 'CUDA Version:'
nvidia-smi | grep 'CUDA Version' | head -1
echo '=========================================='
echo ''
echo 'Loading Python 3.12 environment...'
module load python3
echo ''
echo '🎓 Ready! Run commands with crun:'
echo '   crun -p $ENV_PATH python ~/cuda-lab/verify_cuda.py'
echo '   crun -p $ENV_PATH jupyter lab --no-browser --port=8888'
echo ''
echo 'Multi-GPU example:'
echo '   crun -p $ENV_PATH python -c \"from numba import cuda; print([cuda.gpus[i].name.decode() for i in range(len(cuda.gpus))])\"'
echo ''
cd ~/cuda-lab
exec bash
"
