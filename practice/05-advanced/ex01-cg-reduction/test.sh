#!/bin/bash
echo "Building CG Reduction exercise..."
make clean && make cg_reduction

if [ $? -eq 0 ]; then
    echo ""
    echo "Running tests..."
    ./cg_reduction
    exit $?
else
    echo "Build failed!"
    exit 1
fi
