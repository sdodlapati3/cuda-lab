#!/bin/bash
set -e

echo "Building solution..."
make solution

echo ""
echo "Running solution..."
./solution

echo ""
echo "Test completed!"
