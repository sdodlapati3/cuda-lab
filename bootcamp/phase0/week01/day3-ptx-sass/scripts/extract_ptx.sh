#!/bin/bash
# Extract PTX from compiled binary
#
# PTX is the intermediate representation - like LLVM IR for GPUs

set -e

ARCH=${1:-80}

echo "=== Building for sm_${ARCH} ==="
mkdir -p build
cd build
cmake -G Ninja -DCMAKE_CUDA_ARCHITECTURES=${ARCH} .. > /dev/null 2>&1
ninja > /dev/null 2>&1
cd ..

echo ""
echo "=== Extracting PTX ==="
cuobjdump -ptx build/simple_add > analysis/simple_add_sm${ARCH}.ptx

echo "PTX saved to: analysis/simple_add_sm${ARCH}.ptx"
echo ""
echo "=== PTX Preview (simple_add kernel) ==="
grep -A 50 "\.visible \.entry.*simple_add" analysis/simple_add_sm${ARCH}.ptx | head -60

echo ""
echo "=== Key things to notice ==="
echo "1. Virtual registers: .reg .f32 %f<n> (unlimited)"
echo "2. Memory operations: ld.global, st.global"
echo "3. Predicated branches: @%p1 bra"
echo "4. Thread indexing: %ctaid.x, %tid.x"
