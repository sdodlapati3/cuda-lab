#!/bin/bash
echo "Building Warp-Aggregated Atomics exercise..."
make clean && make warp_atomics

if [ $? -eq 0 ]; then
    echo ""
    echo "Running tests..."
    ./warp_atomics
    exit $?
else
    echo "Build failed!"
    exit 1
fi
