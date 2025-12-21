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
echo "To detect race conditions:"
echo "  compute-sanitizer --tool racecheck ./build/race_example"
echo ""
echo "To verify fixes:"
echo "  compute-sanitizer --tool racecheck ./build/race_fixed"
