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
echo "Run demos:"
echo "  ./build/warp_divergence"
echo "  ./build/simt_demo"
echo ""
echo "Profile divergence:"
echo "  ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./build/warp_divergence"
