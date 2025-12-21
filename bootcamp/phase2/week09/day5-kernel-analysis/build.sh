#!/bin/bash
set -e

ARCH=${1:-80}

mkdir -p build
cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    ..
ninja
cd ..

echo ""
echo "Build complete!"
echo ""
echo "Run kernel analysis:"
echo "  ./build/analyze_kernels"
echo ""
echo "With NCU profiling:"
echo "  ncu --set roofline ./build/analyze_kernels"
