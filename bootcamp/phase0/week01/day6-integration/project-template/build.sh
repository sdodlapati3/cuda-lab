#!/bin/bash
# Quick build script

set -e

ARCH=${1:-80}
BUILD_TYPE=${2:-Release}

echo "Building for sm_${ARCH} (${BUILD_TYPE})"

mkdir -p build
cd build

cmake -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    ..

ninja

echo ""
echo "Build complete!"
echo "  ./build/main       - Run main program"
echo "  ./build/benchmark  - Run benchmarks"
echo "  ./build/test_kernels - Run tests"
