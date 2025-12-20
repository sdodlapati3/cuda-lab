#!/bin/bash
echo "Building CDP Octree exercise..."
make clean && make cdp_octree

if [ $? -eq 0 ]; then
    echo ""
    echo "Running tests..."
    ./cdp_octree
    exit $?
else
    echo "Build failed!"
    exit 1
fi
