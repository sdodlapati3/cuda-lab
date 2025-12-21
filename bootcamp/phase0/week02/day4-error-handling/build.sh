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
echo "To see async error behavior:"
echo "  ./build/async_errors"
echo ""
echo "To see error handling patterns:"
echo "  ./build/error_patterns"
