#!/bin/bash
echo "Building CDP Quicksort exercise..."
make clean && make cdp_quicksort

if [ $? -eq 0 ]; then
    echo ""
    echo "Running tests..."
    ./cdp_quicksort
    exit $?
else
    echo "Build failed!"
    exit 1
fi
