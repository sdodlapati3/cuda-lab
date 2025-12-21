#!/bin/bash
set -e
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
echo "Run with: ./build/naive_gemm [M] [N] [K]"
echo "Default: ./build/naive_gemm 2048 2048 2048"
echo "Profile: ncu --set full ./build/naive_gemm 1024 1024 1024"
