#!/bin/bash
echo "Building CDP Patterns exercise..."
make clean && make cdp_patterns

if [ $? -eq 0 ]; then
    echo ""
    echo "Running tests..."
    ./cdp_patterns
    exit $?
else
    echo "Build failed!"
    exit 1
fi
