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
echo "  ./build/launch_config"
echo "  ./build/occupancy_demo"
echo ""
echo "Profile occupancy:"
echo "  ncu --set full --launch-skip 10 --launch-count 1 ./build/launch_config"
