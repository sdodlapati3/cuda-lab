#!/bin/bash
set -e
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
echo ""
echo "Run directly: ./build/profiling_demo"
echo "Profile with nsys: nsys profile ./build/profiling_demo"
echo "Profile with ncu: ncu --set full ./build/profiling_demo"
