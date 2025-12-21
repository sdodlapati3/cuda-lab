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
echo "Compute profiling commands:"
echo ""
echo "1. Occupancy analysis:"
echo "   ./build/occupancy_test"
echo "   ncu --set compute -o occupancy_report ./build/occupancy_test"
echo ""
echo "2. Warp efficiency:"
echo "   ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./build/warp_efficiency"
echo ""
echo "3. Full compute metrics:"
echo "   ncu --set full ./build/occupancy_test"
