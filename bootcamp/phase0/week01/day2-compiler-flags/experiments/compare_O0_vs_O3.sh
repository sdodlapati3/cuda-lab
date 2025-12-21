#!/bin/bash
# Compare -O0 vs -O3 optimization levels
#
# This demonstrates why you should NEVER use -O0 or -G in production

set -e

echo "=========================================="
echo "  Compiler Optimization Level Comparison"
echo "=========================================="
echo ""

ARCH=${1:-80}

# Build with -O0
echo "Building with -O0 (no optimization)..."
mkdir -p build_O0
cd build_O0
cmake -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    -DCMAKE_BUILD_TYPE=Debug \
    .. > /dev/null 2>&1
ninja > /dev/null 2>&1
cd ..

# Build with -O3
echo "Building with -O3 (maximum optimization)..."
mkdir -p build_O3
cd build_O3
cmake -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    -DCMAKE_BUILD_TYPE=Release \
    .. > /dev/null 2>&1
ninja > /dev/null 2>&1
cd ..

echo ""
echo "=== Running -O0 build ==="
./build_O0/vector_add

echo ""
echo "=== Running -O3 build ==="
./build_O3/vector_add

echo ""
echo "=========================================="
echo "  Key Takeaway: Always use -O3 for"
echo "  performance, -lineinfo for profiling"
echo "=========================================="
