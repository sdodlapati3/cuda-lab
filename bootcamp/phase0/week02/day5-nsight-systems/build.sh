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
echo "Nsight Systems profiling commands:"
echo ""
echo "1. Profile timeline demo:"
echo "   nsys profile -o timeline ./build/timeline_demo"
echo ""
echo "2. Profile bottleneck examples:"
echo "   nsys profile -o bottlenecks ./build/bottleneck_examples"
echo ""
echo "3. View reports:"
echo "   nsys-ui timeline.nsys-rep"
echo "   nsys stats timeline.nsys-rep"
echo ""
echo "4. Full profiling with GPU metrics:"
echo "   nsys profile --trace=cuda,nvtx --cuda-memory-usage=true -o full ./build/timeline_demo"
