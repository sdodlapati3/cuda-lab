#!/bin/bash
# Test script for ex01-device-query

echo "========================================"
echo "Testing Device Query Exercise"
echo "========================================"

# Compile
echo "Compiling solution.cu..."
nvcc solution.cu -o solution 2>&1
if [ $? -ne 0 ]; then
    echo "❌ COMPILATION FAILED"
    exit 1
fi
echo "✅ Compilation successful"

# Run and capture output
echo ""
echo "Running device query..."
./solution > output.txt 2>&1

# Check for expected output patterns
echo ""
echo "Checking output..."

# Must contain these key patterns
patterns=(
    "Device"
    "Compute Capability"
    "Multiprocessors"
    "Global Memory"
    "Shared Memory"
    "Max Threads"
    "Warp Size"
)

all_passed=true
for pattern in "${patterns[@]}"; do
    if grep -qi "$pattern" output.txt; then
        echo "✅ Found: $pattern"
    else
        echo "❌ Missing: $pattern"
        all_passed=false
    fi
done

echo ""
if [ "$all_passed" = true ]; then
    echo "========================================"
    echo "✅ ALL TESTS PASSED!"
    echo "========================================"
    rm -f solution output.txt
    exit 0
else
    echo "========================================"
    echo "❌ SOME TESTS FAILED"
    echo "========================================"
    echo "Your output:"
    cat output.txt
    rm -f solution output.txt
    exit 1
fi
