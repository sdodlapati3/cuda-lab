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
echo "Roofline analysis commands:"
echo ""
echo "1. Run demo with manual roofline analysis:"
echo "   ./build/roofline_demo"
echo ""
echo "2. Profile with ncu roofline view:"
echo "   ncu --set roofline -o roofline_report ./build/roofline_kernels"
echo ""
echo "3. View roofline chart in GUI:"
echo "   ncu-ui roofline_report.ncu-rep"
