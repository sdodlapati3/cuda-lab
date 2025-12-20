#!/bin/bash
echo "Building VMM Growable Buffer exercise..."
make clean && make vmm_growable

if [ $? -eq 0 ]; then
    echo ""
    echo "Running tests..."
    ./vmm_growable
    exit $?
else
    echo "Build failed!"
    exit 1
fi
