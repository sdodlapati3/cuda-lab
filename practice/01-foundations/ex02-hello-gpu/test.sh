#!/bin/bash
# Test script for ex02-hello-gpu

echo "========================================"
echo "Testing Hello GPU Exercise"
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
echo "Running hello GPU..."
./solution > output.txt 2>&1

# Check for expected output patterns
echo ""
echo "Checking output..."

# Should have output from multiple thread configurations
tests_passed=0
tests_total=5

# Test 1: Output exists
if [ -s output.txt ]; then
    echo "✅ Output generated"
    ((tests_passed++))
else
    echo "❌ No output generated"
fi

# Test 2: Contains "Hello"
if grep -qi "hello" output.txt; then
    echo "✅ Contains greeting"
    ((tests_passed++))
else
    echo "❌ Missing greeting output"
fi

# Test 3: Contains block info
if grep -qi "block" output.txt; then
    echo "✅ Contains block information"
    ((tests_passed++))
else
    echo "❌ Missing block information"
fi

# Test 4: Contains thread info  
if grep -qi "thread" output.txt; then
    echo "✅ Contains thread information"
    ((tests_passed++))
else
    echo "❌ Missing thread information"
fi

# Test 5: Multiple configurations run (check for different block counts)
config_count=$(grep -c "===" output.txt)
if [ "$config_count" -ge 3 ]; then
    echo "✅ Multiple configurations tested"
    ((tests_passed++))
else
    echo "❌ Expected 3+ configurations, found $config_count"
fi

echo ""
echo "Score: $tests_passed / $tests_total"
echo ""

if [ "$tests_passed" -eq "$tests_total" ]; then
    echo "========================================"
    echo "✅ ALL TESTS PASSED!"
    echo "========================================"
    rm -f solution output.txt
    exit 0
else
    echo "========================================"
    echo "❌ SOME TESTS FAILED"
    echo "========================================"
    echo ""
    echo "Your output:"
    cat output.txt
    rm -f solution output.txt
    exit 1
fi
