#!/bin/bash
#SBATCH --job-name=gpu-info-b200
#SBATCH --partition=b200flex-8
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu-info-b200-%j.out
#SBATCH --error=logs/gpu-info-b200-%j.err

echo "=== B200 GPU Info ==="
nvidia-smi
echo ""
nvidia-smi -q | grep -E "Product Name|CUDA Version|Driver Version|Total|Bus|Link|Compute Mode|MIG|Memory"
echo ""
nvidia-smi --query-gpu=name,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current,power.limit,clocks.max.sm,clocks.max.memory --format=csv
