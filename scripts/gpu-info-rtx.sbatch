#!/bin/bash
#SBATCH --job-name=gpu-info-rtx
#SBATCH --partition=rtxp6000flex-1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu-info-rtx-%j.out
#SBATCH --error=logs/gpu-info-rtx-%j.err

echo "=== RTX Pro 6000 GPU Info ==="
nvidia-smi
echo ""
nvidia-smi -q | grep -E "Product Name|CUDA Version|Driver Version|Total|Bus|Link|Compute Mode|MIG|Memory"
echo ""
nvidia-smi --query-gpu=name,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current,power.limit,clocks.max.sm,clocks.max.memory --format=csv
