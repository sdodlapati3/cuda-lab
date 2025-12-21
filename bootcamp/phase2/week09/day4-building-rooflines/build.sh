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
echo "Generate roofline data:"
echo "  ./build/roofline_data"
echo ""
echo "Plot roofline (requires matplotlib):"
echo "  python3 plot_roofline.py"
