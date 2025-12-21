#!/bin/bash
set -e

mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja

echo "Running 2D tiling introduction..."
./tiling_intro
