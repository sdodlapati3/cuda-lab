#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-9
# #SBATCH --account=<your_account>  # Uncomment and set your account

# ============================================================================
# Job Array Template for Hyperparameter Sweeps
# ============================================================================
# Use this template for:
# - Hyperparameter tuning
# - Running multiple experiments in parallel
# - Cross-validation folds
# - Ablation studies
# ============================================================================

set -e

echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs results

# Load modules
# module load cuda/12.4
# module load python/3.10
# module load pytorch/2.1

# Activate environment
# source activate myenv

# ============================================================================
# Method 1: Predefined parameter arrays
# ============================================================================

# Define hyperparameter arrays
LEARNING_RATES=(0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.0005 0.005 0.05)
BATCH_SIZES=(16 32 64 128 256 512 1024 32 64 128)
HIDDEN_DIMS=(128 256 512 1024 128 256 512 1024 256 512)

# Get parameters for this task
LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}
BS=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
HD=${HIDDEN_DIMS[$SLURM_ARRAY_TASK_ID]}

echo "Running with: LR=$LR, BS=$BS, HD=$HD"

python train.py \
    --learning-rate $LR \
    --batch-size $BS \
    --hidden-dim $HD \
    --output-dir results/run_${SLURM_ARRAY_TASK_ID}

# ============================================================================
# Method 2: Read from config file
# ============================================================================

# # Assuming configs/sweep.txt has one config per line:
# # --lr 0.001 --bs 32 --hidden 256
# # --lr 0.0001 --bs 64 --hidden 512
# # ...
# 
# CONFIG_FILE="configs/sweep.txt"
# CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $CONFIG_FILE)
# 
# echo "Config: $CONFIG"
# python train.py $CONFIG --output-dir results/run_${SLURM_ARRAY_TASK_ID}

# ============================================================================
# Method 3: Grid search using task ID arithmetic
# ============================================================================

# # For a 5x4 grid of (lr, batch_size):
# NUM_LRS=5
# NUM_BS=4
# 
# LR_IDX=$((SLURM_ARRAY_TASK_ID / NUM_BS))
# BS_IDX=$((SLURM_ARRAY_TASK_ID % NUM_BS))
# 
# LRS=(0.0001 0.001 0.01 0.1 1.0)
# BSS=(16 32 64 128)
# 
# LR=${LRS[$LR_IDX]}
# BS=${BSS[$BS_IDX]}
# 
# python train.py --lr $LR --batch-size $BS

echo ""
echo "=========================================="
echo "Task $SLURM_ARRAY_TASK_ID finished: $(date)"
echo "=========================================="
