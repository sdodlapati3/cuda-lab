#!/bin/bash
set -e

ARCH=${1:-80}

mkdir -p build
cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    ..
ninja
cd ..

echo ""
echo "Build complete!"
echo ""
echo "Run all tests:       ./build/run_tests"
echo "Run filtered tests:  ./build/run_tests -f Reduction"
echo "Run with CTest:      cd build && ctest"
