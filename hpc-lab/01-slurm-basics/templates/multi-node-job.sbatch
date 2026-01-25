#!/bin/bash
#SBATCH --job-name=multi-node
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --exclusive
# #SBATCH --account=<your_account>  # Uncomment and set your account

# ============================================================================
# Multi-Node Distributed Training Job Template
# ============================================================================
# Use this template for:
# - Large-scale distributed training (16+ GPUs)
# - FSDP, DeepSpeed, or Megatron-LM training
# - Production training runs
# ============================================================================

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
echo "Node list: $SLURM_NODELIST"
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

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"

# NCCL settings for multi-node
# ============================================================================
# GCP Waterfield Cluster: a3-highgpu-* instances
# Inter-node: 200 Gbps Ethernet (NO InfiniBand, NO GPUDirect-TCPX)
# Intra-node: NVLink for GPU-to-GPU (fast)
# ============================================================================
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1           # GCP a3-highgpu has no InfiniBand
export NCCL_SOCKET_IFNAME=eth0     # Use Ethernet interface

# Performance tuning for 200 Gbps Ethernet
export NCCL_SOCKET_NTHREADS=4      # More threads for socket operations
export NCCL_NSOCKS_PERTHREAD=2     # Multiple sockets per thread
export NCCL_BUFFSIZE=4194304       # 4MB buffer for better throughput

# For debugging distributed issues
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Print node info
echo ""
echo "Node configuration:"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c 'echo "Node $(hostname): $(nvidia-smi -L)"'
echo ""

# Launch distributed training with srun
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --config config.yaml \
    --output-dir results/$SLURM_JOB_ID \
    --distributed

# Alternative: Using DeepSpeed launcher
# srun deepspeed \
#     --hostfile hostfile.txt \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     train.py \
#     --deepspeed_config ds_config.json

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "=========================================="
