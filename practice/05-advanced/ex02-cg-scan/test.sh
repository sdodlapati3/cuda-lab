#!/bin/bash
echo "Building CG Scan exercise..."
make clean && make cg_scan

if [ $? -eq 0 ]; then
    echo ""
    echo "Running tests..."
    ./cg_scan
    exit $?
else
    echo "Build failed!"
    exit 1
fi
