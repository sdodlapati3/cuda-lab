#!/bin/bash
# Profile with -lineinfo to see source mapping
#
# This shows how Nsight Compute uses -lineinfo to map
# assembly instructions back to your source code

set -e

echo "=========================================="
echo "  Profiling with -lineinfo"
echo "=========================================="
echo ""

ARCH=${1:-80}

# Build with RelWithDebInfo (has -lineinfo)
echo "Building with -O3 -lineinfo..."
mkdir -p build_profile
cd build_profile
cmake -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    .. > /dev/null 2>&1
ninja > /dev/null 2>&1
cd ..

echo ""
echo "To profile with Nsight Compute:"
echo "  ncu --set full -o profile_report ./build_profile/vector_add"
echo ""
echo "To view the report:"
echo "  ncu-ui profile_report.ncu-rep"
echo ""
echo "Or for quick terminal output:"
echo "  ncu --section MemoryWorkloadAnalysis ./build_profile/vector_add"
echo ""

# Run a quick profile if ncu is available
if command -v ncu &> /dev/null; then
    echo "=== Quick Memory Analysis ==="
    ncu --section MemoryWorkloadAnalysis \
        --kernel-name vector_add \
        --launch-skip 5 \
        --launch-count 1 \
        ./build_profile/vector_add 2>&1 | head -50
fi
