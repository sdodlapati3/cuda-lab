#!/bin/bash
# Extract SASS (actual GPU assembly) from compiled binary
#
# SASS is architecture-specific - different for each sm_XX

set -e

ARCH=${1:-80}

echo "=== Building for sm_${ARCH} ==="
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=${ARCH} .. > /dev/null 2>&1
ninja > /dev/null 2>&1
cd ..

mkdir -p analysis

echo ""
echo "=== Extracting SASS ==="
cuobjdump -sass build/simple_add > analysis/simple_add_sm${ARCH}.sass

echo "SASS saved to: analysis/simple_add_sm${ARCH}.sass"

echo ""
echo "=== Register Usage ==="
cuobjdump -res-usage build/simple_add 2>&1 | grep -A2 "Function"

echo ""
echo "=== SASS Preview (simple_add kernel) ==="
grep -A 30 "Function : .*simple_add" analysis/simple_add_sm${ARCH}.sass | head -35

echo ""
echo "=== Key SASS instructions to know ==="
echo "LDG.E    - Global load"
echo "STG.E    - Global store"
echo "FADD     - Float add"
echo "FFMA     - Fused multiply-add"
echo "IMAD     - Integer multiply-add"
echo "S2R      - Special register read (tid, ctaid)"
echo "ISETP    - Integer compare and set predicate"
echo "@P0      - Predicated execution"
