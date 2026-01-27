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

# Check Nsight Systems
echo "=== Nsight Systems ==="
if command -v nsys &> /dev/null; then
    echo "✅ nsys found: $(which nsys)"
    nsys --version
else
    echo "❌ nsys not found in PATH"
    # Search common locations
    for nsys_path in /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems*/bin/nsys; do
        if [ -x "$nsys_path" ]; then
            echo "Found nsys at: $nsys_path"
            $nsys_path --version
            export NSYS_PATH="$nsys_path"
            break
        fi
    done
fi
echo ""

# Check Nsight Compute
echo "=== Nsight Compute ==="
if command -v ncu &> /dev/null; then
    echo "✅ ncu found: $(which ncu)"
    ncu --version
else
    echo "❌ ncu not found in PATH"
    # Search common locations
    for ncu_path in /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute*/ncu; do
        if [ -x "$ncu_path" ]; then
            echo "Found ncu at: $ncu_path"
            $ncu_path --version
            export NCU_PATH="$ncu_path"
            break
        fi
    done
fi
echo ""

# Check ncu-ui (GUI)
echo "=== Nsight Compute GUI ==="
if command -v ncu-ui &> /dev/null; then
    echo "✅ ncu-ui found: $(which ncu-ui)"
else
    echo "ℹ️  ncu-ui not in PATH (GUI typically run locally, not on cluster)"
fi
echo ""

# List all NVIDIA profiling tools found
echo "=== All NVIDIA Tools Found ==="
find /usr/local -name "nsys" -o -name "ncu" -o -name "nvprof" 2>/dev/null | head -20
find /opt -name "nsys" -o -name "ncu" 2>/dev/null | head -10
echo ""

# Quick functionality test
echo "=== Quick Functionality Test ==="
cd $HOME/cuda-lab

# Test nsys with a simple command
if command -v nsys &> /dev/null || [ -n "$NSYS_PATH" ]; then
    NSYS_CMD=${NSYS_PATH:-nsys}
    echo "Testing nsys profile..."
    $NSYS_CMD profile -o /tmp/test_profile --stats=true sleep 1 2>&1 | head -10
    rm -f /tmp/test_profile.nsys-rep
    echo "✅ nsys works!"
else
    echo "⚠️  Cannot test nsys - not found"
fi
echo ""

# Test ncu (may require elevated permissions)
if command -v ncu &> /dev/null || [ -n "$NCU_PATH" ]; then
    NCU_CMD=${NCU_PATH:-ncu}
    echo "Testing ncu (listing available metrics)..."
    $NCU_CMD --list-sets 2>&1 | head -10
    echo "✅ ncu works!"
else
    echo "⚠️  Cannot test ncu - not found"
fi
echo ""

echo "=============================================="
echo "Test Complete"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. If tools are available, build and profile the exercises:"
echo "   cd profiling-lab/02-nsight-compute/ex01-memory-metrics"
echo "   make && ncu --section MemoryWorkloadAnalysis ./memory_bandwidth"
echo ""
echo "2. To generate reports for local viewing:"
echo "   ncu -o report_name --set full ./your_app"
echo "   # Download report_name.ncu-rep to your local machine"
echo "   # Open with ncu-ui"
