#!/bin/bash
#SBATCH --job-name=h100-dev
#SBATCH --partition=h100flex-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

# ============================================================================
# H100 GPU Development Session
# ============================================================================
# This script allocates an H100 GPU node for interactive development via SSH.
#
# Usage:
#   sbatch scripts/h100-dev-session.sbatch
#
# After job starts, check the log file or run:
#   squeue -u $USER
#
# Then SSH to the allocated node:
#   ssh <nodename>
#
# GPU: NVIDIA H100 80GB HBM3 (Hopper Architecture)
# ============================================================================

echo "=============================================="
echo "H100 GPU Development Session Started"
echo "=============================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Job Name:   $SLURM_JOB_NAME"
echo "Node:       $(hostname)"
echo "Start Time: $(date)"
echo "Duration:   4 hours"
echo "=============================================="
echo ""

# Show GPU info
echo "GPU Information:"
echo "----------------"
nvidia-smi -L
echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo ""

# Show CUDA info
echo "CUDA Environment:"
echo "-----------------"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
which nvcc 2>/dev/null && nvcc --version | head -4
echo ""

echo "=============================================="
echo "Ready for SSH connection!"
echo ""
echo "From login node, run:"
echo "  ssh $(hostname)"
echo ""
echo "Then activate environment:"
echo "  module load python3"
echo "  crun -p ~/envs/cuda_lab python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}\")'"
echo ""
echo "To run tests:"
echo "  cd ~/cuda-lab"
echo "  crun -p ~/envs/cuda_lab python -m pytest tests/ -v -m gpu"
echo "=============================================="

# Keep the job running for SSH access
sleep infinity
