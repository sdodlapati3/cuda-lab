#!/bin/bash
set -e
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
echo "Run with: ./build/gemm_setup [M] [N] [K]"
echo "Default: ./build/gemm_setup 4096 4096 4096"
