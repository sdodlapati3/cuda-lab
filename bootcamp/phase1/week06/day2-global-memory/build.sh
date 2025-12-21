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
echo "Run demos:"
echo "  ./build/coalescing_demo"
echo "  ./build/aos_vs_soa"
echo ""
echo "Profile memory efficiency:"
echo "  ncu --set full ./build/coalescing_demo"
