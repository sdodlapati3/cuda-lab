#!/bin/bash
set -e
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja
echo ""
echo "Testing C++ interface:"
./test_vector_ops
echo ""
echo "Python module location (if built):"
ls -la cuda_ops*.so 2>/dev/null || echo "  Python module not built (pybind11 not found)"
