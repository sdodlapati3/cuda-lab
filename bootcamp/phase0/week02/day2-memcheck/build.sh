#!/bin/bash
set -e

ARCH=${1:-80}

echo "Building for sm_${ARCH}..."
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=${ARCH} ..
ninja
cd ..

echo ""
echo "Build complete!"
echo ""
echo "Memory error detection:"
echo "  compute-sanitizer --tool memcheck ./build/memory_errors"
echo ""
echo "Leak detection:"
echo "  compute-sanitizer --tool memcheck --leak-check full ./build/memory_leaks"
echo ""
echo "Verify fixes:"
echo "  compute-sanitizer --tool memcheck ./build/memory_fixed"
