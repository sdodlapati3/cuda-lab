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
echo "Nsight Compute commands:"
echo ""
echo "1. Basic profiling:"
echo "   ncu ./build/vector_ops"
echo ""
echo "2. Full metrics:"
echo "   ncu --set full -o vector_report ./build/vector_ops"
echo ""
echo "3. Specific kernel:"
echo "   ncu --kernel-name vector_add ./build/vector_ops"
echo ""
echo "4. Compare implementations:"
echo "   ncu --set full -o saxpy_report ./build/profile_demo"
echo ""
echo "5. View in GUI:"
echo "   ncu-ui vector_report.ncu-rep"
