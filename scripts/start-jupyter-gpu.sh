#!/bin/bash
#SBATCH --job-name=cuda-jupyter
#SBATCH --partition=t4flex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=jupyter-%j.log

echo "=========================================="
echo "CUDA Learning Session"
echo "=========================================="
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

# Load modules (adjust based on your cluster)
module load cuda 2>/dev/null || echo "No cuda module needed"

# Activate conda environment
source ~/miniconda3/bin/activate cuda-learning 2>/dev/null || \
    source ~/.conda/envs/cuda-learning/bin/activate 2>/dev/null || \
    echo "Activating environment..."

# Get a free port
PORT=$(shuf -i 8000-9000 -n 1)

# Print connection instructions
echo ""
echo "=========================================="
echo "🚀 Jupyter Lab starting on port $PORT"
echo "=========================================="
echo ""
echo "To connect, run this on your LOCAL machine:"
echo ""
echo "  ssh -L $PORT:$(hostname):$PORT $(whoami)@$(hostname -f | sed 's/hpcslurm-nst4flex-[0-9]/LOGIN_NODE/')"
echo ""
echo "Then open in browser: http://localhost:$PORT"
echo ""
echo "=========================================="

# Start Jupyter
cd ~/cuda-lab/learning-path/week-01
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
