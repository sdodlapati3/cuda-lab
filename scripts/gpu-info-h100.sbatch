#!/bin/bash
#SBATCH --job-name=gpu-info-h100
#SBATCH --partition=h100flex-1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu-info-h100-%j.out
#SBATCH --error=logs/gpu-info-h100-%j.err

echo "=== H100 GPU Info ==="
nvidia-smi
echo ""
nvidia-smi -q | grep -E "Product Name|CUDA Version|Driver Version|Total|Bus|Link|Compute Mode|MIG|Memory"
echo ""
nvidia-smi --query-gpu=name,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current,power.limit,clocks.max.sm,clocks.max.memory --format=csv
