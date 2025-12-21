#!/bin/bash
set -e
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
echo "Run with: ./build/particle_system"
