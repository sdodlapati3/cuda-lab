#!/bin/bash
set -e

ARCH=${1:-80}

echo "Building with -G (device debugging)..."
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=${ARCH} ..
ninja
cd ..

echo ""
echo "Build complete!"
echo ""
echo "To debug:"
echo "  cuda-gdb ./build/debug_example"
echo "  cuda-gdb ./build/debug_reduction"
echo ""
echo "Quick reference:"
echo "  break kernel_name   - Set breakpoint"
echo "  run                 - Start program"
echo "  cuda thread (x,y,z) - Switch to thread"
echo "  print var           - Print variable"
echo "  info cuda threads   - List all CUDA threads"
