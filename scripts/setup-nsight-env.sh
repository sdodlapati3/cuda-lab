#!/bin/bash
# =============================================================================
# Nsight Tools Environment Setup
# Source this file before using profiling tools:
#   source scripts/setup-nsight-env.sh
# =============================================================================

# Nsight tools location (installed via conda in nsight_tools env)
export NSIGHT_HOME="$HOME/envs/nsight_tools/nsight-compute-2025.4.1"

# Tool paths
export NSYS="$NSIGHT_HOME/host/target-linux-x64/nsys"
export NCU="$NSIGHT_HOME/ncu"
export NCU_UI="$NSIGHT_HOME/ncu-ui"

# Add to PATH for convenience
export PATH="$NSIGHT_HOME:$NSIGHT_HOME/host/target-linux-x64:$PATH"

# Verify tools are available
echo "=== Nsight Tools Environment ==="
echo "NSIGHT_HOME: $NSIGHT_HOME"
echo ""
echo "Tools available:"
if [ -x "$NSYS" ]; then
    echo "  ✅ nsys:   $NSYS"
    $NSYS --version 2>/dev/null | head -1
else
    echo "  ❌ nsys:   NOT FOUND"
fi

if [ -x "$NCU" ]; then
    echo "  ✅ ncu:    $NCU"
    $NCU --version 2>/dev/null | head -1
else
    echo "  ❌ ncu:    NOT FOUND"
fi

if [ -x "$NCU_UI" ]; then
    echo "  ✅ ncu-ui: $NCU_UI (GUI)"
else
    echo "  ⚠️  ncu-ui: NOT FOUND (GUI - run locally)"
fi
echo ""
echo "Usage examples:"
echo "  nsys profile -o my_report python train.py"
echo "  ncu --set full -o kernel_report ./my_cuda_app"
echo "  ncu-ui my_report.ncu-rep  # Open GUI locally"
echo "================================"
