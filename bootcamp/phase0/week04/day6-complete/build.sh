#!/bin/bash
set -e

ARCH=${1:-80}
BUILD_TYPE=${2:-Release}

mkdir -p build
cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DBUILD_EXAMPLES=ON \
    ..
ninja
cd ..

echo ""
echo "Build complete!"
echo ""
echo "Run application:  ./build/myapp"
echo "Run tests:        ./build/tests"
echo "Run benchmarks:   ./build/benchmarks"
echo "Run example:      ./build/simple_example"
