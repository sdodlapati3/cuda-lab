#!/bin/bash
#SBATCH --job-name=cuda-dev
#SBATCH --partition=rtxp6000flex-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err

# ============================================================================
# GPU Development Session
# ============================================================================
# This script allocates a GPU node for interactive development via SSH.
#
# Usage:
#   sbatch scripts/gpu-dev-session.sbatch
#
# After job starts, check the log file or run:
#   squeue -u $USER
#
# Then SSH to the allocated node:
#   ssh <nodename>
#
# ============================================================================

echo "=============================================="
echo "GPU Development Session Started"
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

# Instructions for user
echo "=============================================="
echo "Ready for SSH connection!"
echo ""
echo "From login node, run:"
echo "  ssh $(hostname)"
echo ""
echo "Then activate environment:"
echo "  module load python3"
echo "  crun -p ~/envs/cuda_lab python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "To run tests:"
echo "  cd ~/cuda-lab"
echo "  crun -p ~/envs/cuda_lab python -m pytest tests/ -m gpu -v"
echo "=============================================="

# Keep the job alive for the full duration
# The job will remain active, allowing SSH access
sleep infinity
