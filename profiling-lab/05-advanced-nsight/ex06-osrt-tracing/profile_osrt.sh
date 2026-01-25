#!/bin/bash
# Profile script with OS runtime tracing enabled
# Usage: ./profile_osrt.sh python your_script.py

set -e

SCRIPT="${1:-python}"
shift 2>/dev/null || true
ARGS="$@"
OUTPUT="osrt_profile"

echo "=================================================="
echo "Nsight Systems with OS Runtime Tracing"
echo "=================================================="

# Full OS runtime tracing profile
nsys profile \
    --trace=cuda,osrt,nvtx \
    --osrt-events=all \
    --sample=cpu \
    --output="$OUTPUT" \
    "$SCRIPT" $ARGS

echo ""
echo "Profile saved to: ${OUTPUT}.nsys-rep"
echo ""
echo "To analyze OS runtime events:"
echo "  nsys stats -r osrtsum ${OUTPUT}.nsys-rep"
echo ""
echo "To view in GUI:"
echo "  nsys-ui ${OUTPUT}.nsys-rep"
