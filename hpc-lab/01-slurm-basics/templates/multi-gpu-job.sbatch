#!/bin/bash
#SBATCH --job-name=multi-gpu
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
# #SBATCH --account=<your_account>  # Uncomment and set your account

# ============================================================================
# Multi-GPU Single Node Training Job Template (DDP)
# ============================================================================
# Use this template for:
# - PyTorch DistributedDataParallel (DDP) training
# - Multi-GPU training on a single node
# - 4-8 GPU training jobs
# ============================================================================

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p logs

# Load modules
# module load cuda/12.4
# module load python/3.10
# module load pytorch/2.1
# module load nccl/2.20

# Activate environment
# source activate myenv

# Set distributed training environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_GPUS_ON_NODE

# NCCL settings for GCP (no InfiniBand)
# Within a single node, NVLink handles GPU-to-GPU communication
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1           # GCP has no InfiniBand
export NCCL_SOCKET_IFNAME=eth0     # Fallback to Ethernet if needed

# Verify GPUs
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Option 1: Using torchrun (recommended for PyTorch 2.0+)
torchrun \
    --standalone \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    train.py \
    --config config.yaml \
    --output-dir results/$SLURM_JOB_ID

# Option 2: Using torch.distributed.launch (legacy)
# python -m torch.distributed.launch \
#     --nproc_per_node=$SLURM_GPUS_ON_NODE \
#     train.py \
#     --config config.yaml

# Option 3: Manual launch (for debugging)
# for i in $(seq 0 $((SLURM_GPUS_ON_NODE-1))); do
#     CUDA_VISIBLE_DEVICES=$i python train.py --local_rank $i &
# done
# wait

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
