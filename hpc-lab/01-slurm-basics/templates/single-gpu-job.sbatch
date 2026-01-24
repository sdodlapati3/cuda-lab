#!/bin/bash
#SBATCH --job-name=single-gpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# #SBATCH --account=<your_account>  # Uncomment and set your account

# ============================================================================
# Single GPU Training Job Template
# ============================================================================
# Use this template for:
# - Single GPU training runs
# - Inference jobs
# - Development/debugging
# ============================================================================

# Exit on error
set -e

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust for your cluster)
# module load cuda/12.4
# module load python/3.10
# module load pytorch/2.1

# Activate your conda environment (if using)
# source activate myenv

# Set CUDA device (usually automatic, but can be explicit)
export CUDA_VISIBLE_DEVICES=0

# Verify GPU is available
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Your training command
# Replace with your actual training script
python train.py \
    --config config.yaml \
    --output-dir results/$SLURM_JOB_ID

# Alternative: Run a CUDA program
# ./my_cuda_program --input data.bin --output result.bin

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
