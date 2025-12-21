#!/bin/bash
set -e

ARCH=${1:-80}

mkdir -p build
cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    ..
ninja 2>&1 | grep -E "(registers|smem|ptxas)"
echo ""
ninja
cd ..

echo ""
echo "Build complete!"
echo "Run: ./build/register_analysis"
