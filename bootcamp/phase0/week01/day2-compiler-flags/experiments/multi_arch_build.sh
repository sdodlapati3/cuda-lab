#!/bin/bash
# Build for multiple GPU architectures
#
# This creates a "fat binary" that runs on V100, A100, and H100

set -e

echo "=========================================="
echo "  Multi-Architecture Build"
echo "=========================================="
echo ""

mkdir -p build_multi
cd build_multi

# Build for V100 (sm_70), A100 (sm_80), and H100 (sm_90)
echo "Configuring for sm_70, sm_80, sm_90..."
cmake -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DCMAKE_BUILD_TYPE=Release \
    -DVERBOSE_PTX=ON \
    ..

echo ""
echo "Building..."
ninja

echo ""
echo "=== Binary size comparison ==="
ls -lh vector_add

echo ""
echo "=== Embedded architectures ==="
cuobjdump -lelf vector_add 2>/dev/null | grep "\.sm_" || echo "Use 'cuobjdump -lelf vector_add' to see"

echo ""
echo "This binary will run on V100, A100, and H100!"
