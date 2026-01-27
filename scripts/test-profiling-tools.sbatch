#!/bin/bash
#SBATCH --job-name=test-profilers
#SBATCH --partition=h100flex-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/profiler-test-%j.out
#SBATCH --error=logs/profiler-test-%j.err

# =============================================================================
# Test Profiling Tools Availability
# Submit with: sbatch scripts/test-profiling-tools.sbatch
# =============================================================================

echo "=============================================="
echo "Profiling Tools Availability Test"
echo "=============================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo ""

# Check GPU
echo "=== GPU Information ==="
nvidia-smi -L
echo ""

# Check CUDA
echo "=== CUDA Installation ==="
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc not found in PATH"
    # Try to find it
    for cuda_path in /usr/local/cuda /usr/local/cuda-* /opt/cuda*; do
        if [ -x "$cuda_path/bin/nvcc" ]; then
            echo "Found nvcc at: $cuda_path/bin/nvcc"
            export PATH="$cuda_path/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
            nvcc --version
            break
        fi
    done
fi
echo ""

# =============================================================================
# Check Nsight Tools from conda environment (Waterfield HPC specific)
# =============================================================================
NSIGHT_HOME="$HOME/envs/nsight_tools/nsight-compute-2025.4.1"
NSYS="$NSIGHT_HOME/host/target-linux-x64/nsys"
NCU="$NSIGHT_HOME/ncu"

echo "=== Nsight Tools (conda env: nsight_tools) ==="
echo "NSIGHT_HOME: $NSIGHT_HOME"
echo ""

# Check Nsight Systems
echo "=== Nsight Systems ==="
if [ -x "$NSYS" ]; then
    echo "✅ nsys found: $NSYS"
    $NSYS --version
    export NSYS_PATH="$NSYS"
elif command -v nsys &> /dev/null; then
    echo "✅ nsys found in PATH: $(which nsys)"
    nsys --version
else
    echo "❌ nsys not found"
    echo "   Expected at: $NSYS"
fi
echo ""

# Check Nsight Compute
echo "=== Nsight Compute ==="
if [ -x "$NCU" ]; then
    echo "✅ ncu found: $NCU"
    $NCU --version
    export NCU_PATH="$NCU"
elif command -v ncu &> /dev/null; then
    echo "✅ ncu found in PATH: $(which ncu)"
    ncu --version
else
    echo "❌ ncu not found"
    echo "   Expected at: $NCU"
fi
echo ""

# Check ncu-ui (GUI)
echo "=== Nsight Compute GUI ==="
NCU_UI="$NSIGHT_HOME/ncu-ui"
if [ -x "$NCU_UI" ]; then
    echo "✅ ncu-ui found: $NCU_UI"
    echo "   (Run locally with X11 forwarding or download reports)"
else
    echo "ℹ️  ncu-ui not found (GUI typically run locally, not on cluster)"
fi
echo ""

# Quick functionality test
echo "=== Quick Functionality Test ==="
cd $HOME/cuda-lab

# Test nsys with a simple command
if [ -n "$NSYS_PATH" ]; then
    echo "Testing nsys profile..."
    $NSYS_PATH profile -o /tmp/test_profile --stats=true sleep 1 2>&1 | head -10
    rm -f /tmp/test_profile.nsys-rep
    echo "✅ nsys works!"
else
    echo "⚠️  Cannot test nsys - not found"
fi
echo ""

# Test ncu (may require elevated permissions)
if [ -n "$NCU_PATH" ]; then
    echo "Testing ncu (listing available metric sets)..."
    $NCU_PATH --list-sets 2>&1 | head -10
    echo "✅ ncu works!"
else
    echo "⚠️  Cannot test ncu - not found"
fi
echo ""

echo "=============================================="
echo "Test Complete"
echo "=============================================="
echo ""
echo "To use these tools, either:"
echo "  1. source scripts/setup-nsight-env.sh"
echo "  2. Use full paths: \$NCU, \$NSYS"
echo ""
echo "Example profiling commands:"
echo "  \$NSYS profile -o my_report python train.py"
echo "  \$NCU --section MemoryWorkloadAnalysis ./my_cuda_app"
echo "  \$NCU -o kernel_report --set full ./my_cuda_app"
echo "2. To generate reports for local viewing:"
echo "   ncu -o report_name --set full ./your_app"
echo "   # Download report_name.ncu-rep to your local machine"
echo "   # Open with ncu-ui"
