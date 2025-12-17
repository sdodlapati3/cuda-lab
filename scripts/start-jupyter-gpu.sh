#!/bin/bash
#SBATCH --job-name=cuda-jupyter
#SBATCH --partition=t4flex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=jupyter-%j.log

ENV_PATH="${ENV_PATH:-$HOME/envs/cuda_lab}"

echo "=========================================="
echo "CUDA Learning Session"
echo "=========================================="
echo "Node: $(hostname)"
echo "Environment: $ENV_PATH"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

# Load Python 3.12 module
module load python3

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
echo "  ssh -L $PORT:$(hostname):$PORT $(whoami)@hpcslurm-slurm-login-001"
echo ""
echo "Then open in browser: http://localhost:$PORT"
echo ""
echo "Job log: jupyter-$SLURM_JOB_ID.log"
echo "=========================================="

# Start Jupyter using crun
cd ~/cuda-lab/learning-path/week-01
crun -p "$ENV_PATH" jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
