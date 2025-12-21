#!/bin/bash
set -e

ARCH=${1:-80}
BUILD_TYPE=${2:-Release}

mkdir -p build
cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    ..
ninja
cd ..

echo ""
echo "Build complete!"
echo "Library: build/libmylib.a"
echo "Example: ./build/example"
