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
echo "  ./build/streams_demo"
echo "  ./build/events_timing"
echo ""
echo "Profile streams:"
echo "  nsys profile --stats=true ./build/streams_demo"
