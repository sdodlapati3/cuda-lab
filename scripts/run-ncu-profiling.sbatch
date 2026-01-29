#!/bin/bash
#SBATCH --job-name=ncu-profile
#SBATCH --partition=h100flex-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/ncu-profile-%j.out
#SBATCH --error=logs/ncu-profile-%j.err

# =============================================================================
# Run Nsight Compute Profiling on Memory Bandwidth Exercise
# Submit with: sbatch scripts/run-ncu-profiling.sbatch
# =============================================================================

echo "=============================================="
echo "Nsight Compute Profiling Job"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# Load python3 for crun
module load python3

# Use nsight_tools conda env which has nvcc and nsight tools
NSIGHT_ENV="$HOME/envs/nsight_tools"
export NSIGHT_HOME="$NSIGHT_ENV/nsight-compute-2025.4.1"
export NCU="$NSIGHT_HOME/ncu"
export NSYS="$NSIGHT_HOME/host/target-linux-x64/nsys"

# Verify tools
echo "=== Tool Versions ==="
crun -p $NSIGHT_ENV nvcc --version | head -4
$NCU --version | head -3
echo ""

# Create output directory for reports
REPORT_DIR="$HOME/cuda-lab/profiling-lab/reports"
mkdir -p $REPORT_DIR
echo "Reports will be saved to: $REPORT_DIR"
echo ""

# Navigate to exercise
cd $HOME/cuda-lab/profiling-lab/02-nsight-compute/ex01-memory-metrics

# Build the exercise
echo "=== Building memory_bandwidth.cu ==="
crun -p $NSIGHT_ENV nvcc -arch=sm_90 -O3 -lineinfo -o memory_bandwidth memory_bandwidth.cu
if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi
echo "✅ Build successful"
echo ""

# Run the program first (without profiling) to see output
echo "=== Running memory_bandwidth (no profiling) ==="
./memory_bandwidth
echo ""

# Profile with Nsight Compute - Memory Analysis
echo "=== Profiling with Nsight Compute (Memory Analysis) ==="
$NCU --set full \
    -o $REPORT_DIR/memory_bandwidth_report \
    --force-overwrite \
    ./memory_bandwidth

echo ""
echo "=== Profiling Complete ==="
echo ""

# List generated reports
echo "=== Generated Reports ==="
ls -la $REPORT_DIR/
echo ""

# Also run Nsight Systems for timeline
echo "=== Running Nsight Systems Timeline ==="
$NSYS profile -o $REPORT_DIR/memory_bandwidth_timeline ./memory_bandwidth
echo ""

echo "=============================================="
echo "Reports saved to: $REPORT_DIR"
echo "=============================================="
echo ""
echo "To download and view:"
echo "  1. From your local machine:"
echo "     scp <user>@waterfield.odu.edu:~/cuda-lab/profiling-lab/reports/*.ncu-rep ."
echo "     scp <user>@waterfield.odu.edu:~/cuda-lab/profiling-lab/reports/*.nsys-rep ."
echo ""
echo "  2. Open with Nsight tools locally:"
echo "     ncu-ui memory_bandwidth_report.ncu-rep"
echo "     nsys-ui memory_bandwidth_timeline.nsys-rep"
