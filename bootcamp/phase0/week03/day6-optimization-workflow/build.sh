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
echo "Run the complete optimization workflow:"
echo "  ./build/optimization_workflow"
echo ""
echo "Profile each version:"
echo "  ncu --kernel-name reduce_v0 --set full ./build/optimization_workflow"
echo "  ncu --kernel-name reduce_v5 --set full ./build/optimization_workflow"
echo ""
echo "Compare versions in ncu-ui:"
echo "  ncu --set full -o reduction_versions ./build/optimization_workflow"
echo "  ncu-ui reduction_versions.ncu-rep"
