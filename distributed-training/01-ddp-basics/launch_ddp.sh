#!/bin/bash
# launch_ddp.sh - Launch scripts for DDP training
#
# Usage:
#   ./launch_ddp.sh single <script.py> [args...]  # Single node
#   ./launch_ddp.sh slurm <script.py> [args...]   # SLURM cluster
#
# Examples:
#   ./launch_ddp.sh single ddp_mnist.py --epochs 10
#   ./launch_ddp.sh slurm ddp_training.py --batch-size 64

set -e

MODE="${1:-single}"
SCRIPT="${2:-ddp_mnist.py}"
shift 2 2>/dev/null || true
ARGS="$@"

# Get directory of this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# Configuration
# ============================================================================

# Default values (override with environment variables)
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ============================================================================
# Functions
# ============================================================================

launch_single() {
    # Single node, multiple GPUs
    echo "==========================================="
    echo "Launching DDP on single node"
    echo "==========================================="
    echo "GPUs: $GPUS_PER_NODE"
    echo "Script: $SCRIPT"
    echo "Args: $ARGS"
    echo "==========================================="
    
    torchrun \
        --standalone \
        --nproc_per_node=$GPUS_PER_NODE \
        "$SCRIPT_DIR/$SCRIPT" $ARGS
}

launch_slurm() {
    # Multi-node with SLURM
    echo "==========================================="
    echo "Launching DDP with SLURM"
    echo "==========================================="
    echo "Nodes: $NUM_NODES"
    echo "GPUs per node: $GPUS_PER_NODE"
    echo "Script: $SCRIPT"
    echo "Args: $ARGS"
    echo "==========================================="
    
    # Create a temporary SLURM script
    SLURM_SCRIPT=$(mktemp /tmp/ddp_slurm.XXXXXX.sh)
    
    cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ddp_training
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=$GPUS_PER_NODE
#SBATCH --cpus-per-task=$((GPUS_PER_NODE * 4))
#SBATCH --time=04:00:00
#SBATCH --output=ddp_%j.out
#SBATCH --error=ddp_%j.err

# Load modules (customize for your cluster)
# module load cuda/12.0
# module load nccl/2.18

# Set environment variables
export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=\$((SLURM_NNODES * $GPUS_PER_NODE))

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

echo "Master: \$MASTER_ADDR:\$MASTER_PORT"
echo "World size: \$WORLD_SIZE"

# Launch with srun
srun --kill-on-bad-exit=1 bash -c '
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    
    python -m torch.distributed.run \\
        --nnodes=$NUM_NODES \\
        --nproc_per_node=$GPUS_PER_NODE \\
        --node_rank=\$SLURM_NODEID \\
        --master_addr=\$MASTER_ADDR \\
        --master_port=\$MASTER_PORT \\
        $SCRIPT_DIR/$SCRIPT $ARGS
'
EOF

    echo "Submitting SLURM job..."
    sbatch "$SLURM_SCRIPT"
    echo "SLURM script saved to: $SLURM_SCRIPT"
}

launch_manual() {
    # Manual multi-node (provide MASTER_ADDR, NODE_RANK)
    if [ -z "$MASTER_ADDR" ] || [ -z "$NODE_RANK" ]; then
        echo "Error: MASTER_ADDR and NODE_RANK must be set for manual mode"
        echo "Usage: MASTER_ADDR=10.0.0.1 NODE_RANK=0 NUM_NODES=2 ./launch_ddp.sh manual script.py"
        exit 1
    fi
    
    echo "==========================================="
    echo "Launching DDP manually"
    echo "==========================================="
    echo "Master: $MASTER_ADDR:$MASTER_PORT"
    echo "Node rank: $NODE_RANK / $NUM_NODES"
    echo "GPUs: $GPUS_PER_NODE"
    echo "Script: $SCRIPT"
    echo "==========================================="
    
    torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        "$SCRIPT_DIR/$SCRIPT" $ARGS
}

# ============================================================================
# Main
# ============================================================================

case "$MODE" in
    single)
        launch_single
        ;;
    slurm)
        launch_slurm
        ;;
    manual)
        launch_manual
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 <single|slurm|manual> <script.py> [args...]"
        exit 1
        ;;
esac
