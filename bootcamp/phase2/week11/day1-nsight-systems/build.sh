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
echo "Run and profile:"
echo "  ./build/timeline_demo"
echo "  nsys profile -o timeline ./build/timeline_demo"
echo "  nsys stats timeline.nsys-rep"
