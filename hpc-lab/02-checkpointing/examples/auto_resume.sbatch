#!/bin/bash
#SBATCH --job-name=train_auto_resume
#SBATCH --account=<your_account>
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# =============================================================================
# Auto-Resume Training Script
#
# This script automatically resumes training from the latest checkpoint when
# requeued due to time limits or preemption.
#
# Features:
# - Automatic checkpoint detection and resume
# - Signal handling for graceful shutdown
# - Requeue on timeout
#
# Usage:
#   sbatch auto_resume.sbatch
#
# To manually requeue:
#   scontrol requeue <job_id>
# =============================================================================

# Configuration
CHECKPOINT_DIR="./checkpoints"
SCRIPT_PATH="./train_with_checkpoint.py"
LOG_DIR="./logs"

# Create directories
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

# Load modules (adjust for your system)
module load pytorch
module load cudatoolkit

# Activate conda environment (if applicable)
# source activate myenv

# Print job info
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Time: $(date)"
echo "========================================"

# Detect if this is a restart
if [ -f "${CHECKPOINT_DIR}/latest.pt" ]; then
    echo "Found existing checkpoint - resuming training"
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found - starting fresh"
    RESUME_FLAG=""
fi

# Handle SIGTERM for graceful shutdown
# When Slurm sends SIGTERM (before SIGKILL), save checkpoint
handle_sigterm() {
    echo "Received SIGTERM - allowing time for checkpoint save..."
    # The Python script handles the signal; we just wait
    wait ${TRAIN_PID}
    exit 0
}
trap handle_sigterm SIGTERM

# Handle timeout requeue
handle_timeout() {
    echo "Received timeout signal - requeuing job..."
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap handle_timeout SIGUSR1

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))

echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "World size: ${WORLD_SIZE}"

# Enable NCCL debugging (optional)
# export NCCL_DEBUG=INFO

# Run training
srun --kill-on-bad-exit=1 \
    python ${SCRIPT_PATH} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --checkpoint-every 100 \
    --epochs 100 \
    --batch-size 32 \
    ${RESUME_FLAG} &

TRAIN_PID=$!
wait ${TRAIN_PID}
EXIT_CODE=$?

echo "Training finished with exit code: ${EXIT_CODE}"
echo "Time: $(date)"

# If training completed successfully, don't requeue
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training exited with error code ${EXIT_CODE}"
    # Optionally requeue on failure
    # scontrol requeue ${SLURM_JOB_ID}
fi

exit ${EXIT_CODE}
