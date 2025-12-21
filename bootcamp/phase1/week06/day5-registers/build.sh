#!/bin/bash
set -e

ARCH=${1:-80}

mkdir -p build
cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    ..
ninja 2>&1 | grep -E "(registers|lmem|smem|ptxas)" || true
cd ..

echo ""
echo "Build complete! (Register usage shown above)"
echo ""
echo "Run demos:"
echo "  ./build/register_demo"
echo "  ./build/spill_demo"
