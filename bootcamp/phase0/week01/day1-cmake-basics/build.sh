#!/bin/bash
# Build script for Day 1: CMake basics
#
# Usage:
#   ./build.sh              # Default build (sm_80)
#   ./build.sh 90           # Build for H100 (sm_90)
#   ./build.sh 86           # Build for RTX 3090 (sm_86)

set -e  # Exit on error

ARCH=${1:-80}  # Default to sm_80 (A100)

echo "=== Building for sm_${ARCH} ==="

# Create build directory
mkdir -p build
cd build

# Configure with CMake using Ninja generator
cmake -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    -DCMAKE_BUILD_TYPE=Release \
    ..

# Build
ninja

echo ""
echo "=== Build complete! ==="
echo "Run with: ./build/hello_gpu"
