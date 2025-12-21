#!/bin/bash
# Compare SASS across different GPU architectures
#
# Shows how the same kernel compiles differently for different GPUs

set -e

echo "=========================================="
echo "  SASS Comparison Across Architectures"
echo "=========================================="

mkdir -p analysis

# Build for each architecture
for ARCH in 70 80 90; do
    echo ""
    echo "=== Building for sm_${ARCH} ==="
    mkdir -p build_${ARCH}
    cd build_${ARCH}
    cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=${ARCH} .. > /dev/null 2>&1
    ninja > /dev/null 2>&1
    cd ..
    
    # Extract SASS
    cuobjdump -sass build_${ARCH}/simple_add > analysis/sass_sm${ARCH}.txt 2>/dev/null || true
    
    # Count instructions
    INSTR_COUNT=$(grep -E "^\s+/\*[0-9a-f]+\*/" analysis/sass_sm${ARCH}.txt 2>/dev/null | wc -l || echo "0")
    echo "sm_${ARCH}: ${INSTR_COUNT} instructions"
done

echo ""
echo "=== Register Usage Comparison ==="
for ARCH in 70 80 90; do
    REG_COUNT=$(cuobjdump -res-usage build_${ARCH}/simple_add 2>/dev/null | grep "REG" | head -1 | awk '{print $NF}' || echo "?")
    echo "sm_${ARCH}: ${REG_COUNT} registers per thread"
done

echo ""
echo "=== Instruction Differences ==="
echo "To see detailed differences:"
echo "  diff analysis/sass_sm70.txt analysis/sass_sm80.txt"
echo "  diff analysis/sass_sm80.txt analysis/sass_sm90.txt"
