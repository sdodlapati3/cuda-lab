#!/bin/bash
set -e

mkdir -p build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja

echo ""
echo "Build complete. Run with: ./build/model_loading"
