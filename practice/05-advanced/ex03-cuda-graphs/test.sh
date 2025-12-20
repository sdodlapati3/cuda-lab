#!/bin/bash
echo "Building CUDA Graphs exercise..."
make clean && make cuda_graphs

if [ $? -eq 0 ]; then
    echo ""
    echo "Running tests..."
    ./cuda_graphs
    exit $?
else
    echo "Build failed!"
    exit 1
fi
