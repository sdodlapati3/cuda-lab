#!/bin/bash
set -e
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
./tensor_core_intro
