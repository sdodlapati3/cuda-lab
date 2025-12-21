#!/bin/bash
set -e
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
echo ""
echo "Build complete. Run with: ./build/capstone [problem_size]"
echo "Profile with: nsys profile ./build/capstone"
