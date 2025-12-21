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
echo "Profile commands:"
echo "  ncu --set default ./build/ncu_demo"
echo "  ncu --set full ./build/ncu_demo"
echo "  ncu --set roofline ./build/ncu_demo"
