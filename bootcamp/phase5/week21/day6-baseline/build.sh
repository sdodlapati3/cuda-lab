#!/bin/bash
set -e

mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja

echo "Running Week 21 baseline comparison..."
./baseline
